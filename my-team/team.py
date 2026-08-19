"""Your team. Edit this file.

Everything you write goes in here. A submission is one file of code — extra
.py files next to it are not importable when the server loads your team, so
they are rejected rather than silently ignored.

This is a working team, so you can play it right now and see numbers come back:

    START.cmd play my-team --against balanced      (Windows)
    ./start.sh play my-team --against balanced     (macOS and Linux)

It is about as simple as a team can be while still playing football. Every
player asks two questions and does one of four things:

    is the ball ours?
        am I the closest of us to it?   ->  shoot at their goal
        otherwise                       ->  push up towards their goal
    otherwise
        am I the closest of us to it?   ->  go and get it
        otherwise                       ->  drop back towards our goal

It will lose to the reference opponents, which is the point — it is something
to improve, not something to hand in as it stands. Things wrong with it, in
roughly the order they cost you goals: nobody keeps goal, the four players who
are not chasing all run to the same place, nobody ever passes, and every shot
is struck at full power from wherever the player is standing.

The other half of your submission is `team.toml` next to this file, which says
who you are. Fill that in too.

You write one method, `act`, and it is called on every tick of the match —
whether the ball is yours or theirs. Deciding which of those it is, and what
your players should be doing about it, is the assignment. There is no engine
call that will do it for you: an observation reports where everybody is and how
the ball behaves, and the football is yours to write.

You are always given the same pitch whichever side you are really playing:
your goal at the left, theirs at the right, forward is +x, your players are
ids 0..n-1. Write one team, not one per side.

The full API is in the student guide that came with this download.
"""

from soccer import PlayerAction, TeamAction, TeamController, direction


class MyTeam(TeamController):
    # Put your own name and student number here.
    name = "my_team"
    version = "1"

    def initial_formation(self, field):
        """Where your team stands at every kickoff. Optional — delete for the default.

        One (x, y) per player in slot order, slot 0 the goalkeeper. It has to be
        inside the pitch, on your own half and outside the centre circle; a spot
        that is not gets moved to the nearest one that is, rather than refused.
        """
        keeper_x = field.my_goal[0] + field.player_radius * 2
        # Just outside the centre circle: as close to the ball as the rules
        # allow, which is where the kickoff is won or lost.
        forward_x = -field.centre_circle_radius - 1.0
        return [
            (keeper_x, 0.0),        # keeper
            (-20.0, -12.0),         # left back
            (-20.0, 12.0),          # right back
            (forward_x, -6.0),      # left forward
            (forward_x, 6.0),       # right forward
        ]

    def act(self, obs):
        """Called on every tick. Return one action per player.

        The first thing this does is work out whose ball it is, because almost
        everything else follows from that. `obs.ball.controlling_team == 0` is
        the engine's own reading of possession, and it persists while the ball
        runs loose. `closest_to_ball(obs)` — import it from `soccer` — is the
        other way to ask, and it flips the moment somebody outruns you. They
        disagree often, and choosing between them is your first real decision.

        Your split does not have to be this one, either. It can be per player,
        or by where the ball is on the pitch, or something else entirely.
        """
        actions = TeamAction()
        ours = obs.ball.controlling_team == 0
        chaser = obs.closest_my_player_to(obs.ball.position)

        for player in obs.my_players:
            nearest = player.id == chaser.id

            if ours and nearest:
                # Ours, and I am the one on it: have a go at goal. Check
                # can_kick first — a kick from out of range is a wasted tick,
                # and comes back to you as a "kick_rejected" event.
                if obs.can_kick(player.id):
                    actions.set(
                        player.id,
                        PlayerAction(
                            movement=direction(player.position, obs.ball.position),
                            kick_direction=direction(player.position, obs.opponent_goal),
                            kick_power=1.0,
                        ),
                    )
                else:
                    actions.move(
                        player.id, direction(player.position, obs.ball.position)
                    )

            elif ours:
                # Ours, and somebody else is on it: get up the pitch. Running
                # at the goal puts all four of you in the same place, which is
                # the first thing worth fixing — nobody can receive a pass
                # standing on top of the carrier.
                actions.move(
                    player.id, direction(player.position, obs.opponent_goal)
                )

            elif nearest:
                # Theirs, and I am the closest: go and win it back.
                actions.move(player.id, direction(player.position, obs.ball.position))

            else:
                # Theirs, and somebody else is closer: get behind the ball.
                actions.move(player.id, direction(player.position, obs.field.my_goal))

        return actions
