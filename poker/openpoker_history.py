"""Reconstruct public betting history from sequenced OpenPoker events.

Replay is validated against authoritative snapshot stacks/bets/folds/pot/actor.
Sequence jumps are normal. Missing hand starts, incomplete replay and ambiguous
chip normalization yield an explicit unavailable result, never a fabricated key.
"""
from collections import OrderedDict
import math

from .fast_game import FastGame

STREETS={'preflop':0,'flop':1,'turn':2,'river':3}


class PublicHistory:
    def __init__(self, capacity=4096):
        self.capacity=capacity
        self.table_id=None
        self.events=OrderedDict()
        self.conflicts=set()
        self.watermark=0
        self.last_reason='no history'

    def ingest(self, event):
        table=event.get('table_id')
        if not table: return
        if table!=self.table_id:
            self.table_id=table; self.events.clear(); self.conflicts.clear(); self.watermark=0
        seq=event.get('table_seq')
        if type(seq) is not int: return
        self.watermark=max(self.watermark,seq)
        if event.get('type') not in ('hand_start','player_action'): return
        hand=event.get('hand_id')
        if not hand: return
        # Ignore timestamp differences when comparing replay duplicates.
        data={k:v for k,v in event.items() if k not in ('ts','stream','state_hash')}
        existing=self.events.get(seq)
        if existing is not None and existing!=data:
            self.conflicts.add(hand)
        else:
            self.events[seq]=data
        if len(self.events)>self.capacity:
            # Evict by sequence, not arrival order (replay can arrive out of order).
            del self.events[min(self.events)]

    def resync(self, message):
        def sequence(event):
            value=event.get('table_seq')
            return value if type(value) is int else -1
        for event in sorted(message.get('replayed_events',[]),key=sequence):
            self.ingest(event)
        return self.reconstruct(message['snapshot'],message.get('to_table_seq'))

    def reconstruct(self, snapshot, upper_seq=None):
        try:
            history=self._reconstruct(snapshot,upper_seq)
            self.last_reason='verified'
            return history,True
        except (ValueError,KeyError,TypeError,IndexError,StopIteration) as exc:
            self.last_reason=str(exc)
            return (),False

    def _reconstruct(self, snapshot, upper_seq):
        if snapshot.get('table_id')!=self.table_id: raise ValueError('table history unavailable')
        hand=snapshot['hand_id']
        if hand in self.conflicts: raise ValueError('conflicting duplicate history event')
        upper_seq=snapshot.get('table_seq',upper_seq)
        if type(upper_seq) is not int: raise ValueError('snapshot watermark unavailable')
        events=[e for seq,e in sorted(self.events.items()) if seq<=upper_seq and e['hand_id']==hand]
        starts=[e for e in events if e['type']=='hand_start']
        if len(starts)!=1: raise ValueError('complete hand start unavailable')
        start=starts[0]
        if start['dealer_seat']!=snapshot['dealer_seat']: raise ValueError('dealer mismatch')
        bb=snapshot['big_blind']; sb=start['blinds']['small_blind']; scale=bb/2
        if type(bb) is not int or type(sb) is not int or bb<=0 or sb<=0 or bb!=2*sb:
            raise ValueError('unsupported blinds')
        def chips(value):
            if type(value) is not int or value<0:
                raise ValueError('invalid chip amount')
            return value
        unit=lambda value:round(value/scale)
        seats=sorted([s for s in snapshot['seats'] if s.get('in_hand')],key=lambda s:s['seat'])
        dealer=snapshot['dealer_seat']
        seats=[s for s in seats if s['seat']>dealer]+[s for s in seats if s['seat']<=dealer]
        if not 3<=len(seats)<=6: raise ValueError('unsupported hand size')
        mapping={s['seat']:i for i,s in enumerate(seats)}
        actions=[e for e in events if e['type']=='player_action' and e['table_seq']>start['table_seq']]
        contributions={s['seat']:0 for s in seats}
        contributions[seats[0]['seat']]=sb; contributions[seats[1]['seat']]=bb
        pot=sb+bb; replay=[]
        for event in actions:
            seat=event['seat']
            after=chips(event['pot_after'] if 'pot_after' in event else event['pot'])
            delta=after-pot
            unapplied_all_in=(event.get('action')=='all_in' and delta==0
                              and event.get('contribution_delta') in (0,None)
                              and chips(event.get('stack_after') if 'stack_after' in event else event['stack'])>0)
            if unapplied_all_in:
                delta=chips(event.get('stack_after') if 'stack_after' in event else event['stack'])
                after=pot+delta
            if delta<0: raise ValueError('pot moved backwards')
            if event.get('pot_before') is not None and chips(event['pot_before'])!=pot:
                raise ValueError('missing action before event')
            if not unapplied_all_in and event.get('contribution_delta') is not None and chips(event['contribution_delta'])!=delta:
                raise ValueError('contribution mismatch')
            contributions[seat]+=delta
            replay.append((event,delta,pot,unapplied_all_in))
            pot=after
        if pot!=chips(snapshot['pot']): raise ValueError('incomplete pot history')
        initial=[chips(s['stack'])+contributions[s['seat']] for s in seats]
        pad=6-len(seats)
        initial=initial[:2]+[100*bb]*pad+initial[2:]
        game=FastGame(0,6,button=5,stacks=initial)
        # Reinitialize blinds in raw chip units before replaying actions.
        game.stacks=initial[:]
        game.bets=[0]*6
        for i,blind in ((0,sb),(1,bb)):
            amount=min(blind,game.stacks[i])
            game.bets[i]=amount; game.stacks[i]-=amount
        game.paid=game.bets[:]
        game._begin(2)
        history=[]
        # Match the adapter's existing short-handed-to-six-max embedding.
        for i in range(pad):
            o=game.observe()
            history.append((-1,o.position,0,'fold',unit(o.pot),unit(o.call)))
            game.act('fold')
        for event,delta,before,unapplied_all_in in replay:
            if not game.live: raise ValueError('action after terminal state')
            pos=mapping[event['seat']]
            pos=pos+pad if pos>=2 else pos
            reported_street=STREETS.get(event['street'], 4 if event['street']=='showdown' else -1)
            street=game.street
            if game.actors[0]!=pos or sum(game.paid)!=before:
                raise ValueError('missing or out-of-order public action')
            owed=min(game.stacks[pos],max(game.bets)-game.bets[pos])
            kind=event['action']
            if kind in ('check','call') or kind=='all_in' and delta==owed and delta==game.stacks[pos]:
                if delta!=owed or kind=='check' and owed: raise ValueError('call/check contribution mismatch')
                raw_action=norm_action='call'
            elif kind=='fold':
                if delta: raise ValueError('fold contributed chips')
                raw_action=norm_action='fold'
            elif kind in ('raise','all_in'):
                raw_action=game.bets[pos]+delta
                top=max(game.bets)
                lo=top+max(bb,game.raise_size)
                if (not game.can_raise() or delta>game.stacks[pos] or raw_action<=top
                        or raw_action<lo and delta!=game.stacks[pos]):
                    raise ValueError('illegal raise in reconstructed history')
                if kind=='all_in' and delta!=game.stacks[pos]: raise ValueError('incomplete all-in')
                norm_action=max(unit(top)+1,unit(raw_action))
            else: raise ValueError('unknown public action')
            if event.get('to_call_before') is not None and chips(event['to_call_before'])!=owed:
                raise ValueError('call price mismatch')
            history.append((event['seat'],pos,street,norm_action,unit(before),unit(owed)))
            game.act(raw_action,False)
            # Live events label a round-closing action with the resulting street.
            # Older fixtures label the acting street. Accept either only when the
            # validated ledger actually produces that transition.
            resulting_street=game.street if game.live else 4
            if reported_street not in (street, resulting_street):
                raise ValueError('action street disagrees with reconstructed transition')
            expected_stack=0 if unapplied_all_in else chips(event['stack_after'] if 'stack_after' in event else event['stack'])
            if expected_stack!=game.stacks[pos]:
                raise ValueError('stack history mismatch')
        if not game.live: raise ValueError('snapshot is not an acting hand')
        street=STREETS[snapshot['street']] if isinstance(snapshot.get('street'),str) else snapshot.get('street',len(snapshot['board'])-2 if snapshot['board'] else 0)
        if game.street!=street: raise ValueError('street mismatch')
        for s in seats:
            pos=mapping[s['seat']]; pos=pos+pad if pos>=2 else pos
            if (game.stacks[pos]!=chips(s['stack']) or game.bets[pos]!=chips(s['bet'])
                    or game.active[pos] is not (not s['folded'])):
                raise ValueError('snapshot disagrees with reconstructed stacks/bets/folds')
        actor=mapping[snapshot['actor_seat']]; actor=actor+pad if actor>=2 else actor
        if game.actors[0]!=actor: raise ValueError('snapshot actor mismatch')
        return tuple(history)
