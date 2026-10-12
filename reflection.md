# Task 3 Reflection: The AI Paradoxes

Working on this homework — building a CV verifier in Task 1 and then defeating
five such verifiers in Task 2 — made several paradoxes of practical AI feel
less like philosophy and more like engineering facts.

**The automation paradox.** The better an automated verifier gets, the more
humans trust it and the less they supervise it — so when it finally fails, the
failure does more damage. My Task 1 agent scores 7/7 on the public set and
produces confident numbers like 0.97, yet a one-page PDF with an invisible
paragraph can flip its verdict. High average accuracy is exactly what makes
the residual error dangerous: nobody double-checks a system that is "always
right", so its rare mistakes pass straight into KYC and hiring decisions.

**The asymmetry paradox (attack is cheaper than defence).** Defending the
verifier required explicit instructions about untrusted data, careful wording,
and even then the defence is a soft prompt, not a guarantee. Attacking it
required only a text layer the human eye cannot see. Because the same model
reads both instructions and data, the instruction hierarchy is a convention the
model tries to maintain, not a security boundary. Every defence I can write in
a system prompt, a better injection can argue with — the arms race is
structurally tilted towards the attacker.

**The manufactured-trust paradox.** The tool exists to create trust (an
automated KYC check), yet trust itself becomes an attack surface. My fake
"Background check passed — Ref BG-2026-88417" badge costs nothing to print but
borrows the authority of the entire auditing industry. An agent that defers to
authority cues ends up certifying its own deception: the more convincing the
trust signal, the less verification actually happens.

**The delegation paradox.** Verifying a CV is, for a human, a job of
suspicion: reading between the lines, noticing a date that "feels" shifted.
This is precisely what current AI is worst at and what we are fastest to
delegate. Meanwhile the tasks agents do well — exhaustive field-by-field
comparison — humans find boring and error-prone. So we automate the part we
were good at and keep the weaknesses of both sides: machines that can be
talked out of their conclusions by a hidden sentence, and humans who have
forgotten how to spot a lie.

The deepest lesson of the homework is that a reliability score is not evidence
of truth — it is a prediction made by a system whose inputs can be adversarial.
Any deployment of agentic AI against dishonest humans must assume the inputs
are hostile and treat the model's confidence as untrusted as the documents it
reads.
