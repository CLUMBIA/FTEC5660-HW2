# Task 3 Reflection: the AI paradoxes

After building a CV verifier in Task 1 and then breaking five similar verifiers
in Task 2, one paradox stood out to me much more than the others.

**Attacking is always cheaper than defending.** This is not new to AI - it is
the same reason cybersecurity feels like a losing battle. A company has to
defend thousands of servers 24/7 and block every single attack, while an
attacker only has to succeed once, and the payoff (money, data) can be huge
compared to the cost of trying. So attackers happily try a hundred cheap tricks
and lose 99 of them; defenders cannot afford a single miss.

This homework showed the exact same maths in miniature. Defending my verifier
took a carefully written system prompt, explicit warnings that CV text and tool
outputs are untrusted data, plus retries and fallbacks so the pipeline never
crashes. Attacking it took one paragraph of invisible text hidden in a PDF -
something anyone can type in five minutes - and that was enough to potentially
fool an agent that otherwise works well. The same model reads both the
instructions and the data, so the "instruction hierarchy" is really just a
convention the model tries to follow, not a wall. Every defence I write in a
prompt, a better injection can argue with.

A second, smaller observation: fake trust is cheap to manufacture. My CV just
printed "PASS - Background check passed" and suddenly it looked pre-verified.
We see the same thing everywhere in daily life - fake Taobao reviews, a
high-vis vest that makes people assume you belong somewhere. The more a system
(or a person) relies on surface trust signals, the easier it is to cheat.

So the lesson I take away: an AI verifier's confidence score is not evidence of
truth. It is just the model's opinion about text that might be trying to
manipulate it.
