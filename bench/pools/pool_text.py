"""Seeded English prose and source code for the pools (internal to build_pools).

Interface:
  Writer(rng, domains)            one text source; `domains` picks which topic
                                  banks it may draw on, so two writers given
                                  disjoint domain sets cannot produce the same
                                  paragraph
  Writer.document(min_chars)      a document (article, minutes, email thread,
                                  FAQ, story or report) of at least min_chars
  Writer.paragraph()              one paragraph of prose
  Writer.code(min_chars)          source code (Python, JavaScript, SQL, C, shell)
  Writer.python_module(n_funcs)   (source, [function names]) for code completion
  Writer.record()                 (text, facts dict) for JSON extraction
  Writer.name()                   a person's name

Everything is written for this repo; nothing is copied from any source. Only
`random.Random` draws are used (no set iteration, no hash()), so one seed gives
one text on every machine.
"""
from __future__ import annotations

import random
import re

# ---------------------------------------------------------------------------
# Topic banks. Each domain is wholly separate from the others; a Writer only
# sees the domains it was given.
# ---------------------------------------------------------------------------
# verbs are (base, third person, past, -ing)
def _v(*items):
    return [tuple(x.split("/")) for x in items]


DOMAINS: list[dict] = [
    dict(name="harbour",
         actors=["the harbour authority", "a team of pilots", "the night shift supervisor",
                 "the port engineer", "two customs inspectors", "the tug crew", "a local shipping agent",
                 "the dredging contractor"],
         objects=["the tidal gauge", "a set of mooring lines", "the container crane", "the pilot boat",
                  "the quay wall", "the berth schedule", "the fuel barge", "the harbour chart",
                  "the lock gates", "the cargo manifest"],
         verbs=_v("inspect/inspects/inspected/inspecting", "recalibrate/recalibrates/recalibrated/recalibrating",
                  "reschedule/reschedules/rescheduled/rescheduling", "document/documents/documented/documenting",
                  "repair/repairs/repaired/repairing", "audit/audits/audited/auditing"),
         places=["Berth Nine", "the outer breakwater", "the eastern terminal", "the pilot station",
                 "the customs shed", "the fuel jetty"],
         facts=["Spring tides at the port rise almost a metre higher than neap tides.",
                "A loaded container ship needs several kilometres of open water to stop.",
                "Mooring lines are replaced on a fixed schedule, whether or not they look worn.",
                "Silt builds up fastest in the weeks after heavy river flooding.",
                "Pilots board incoming vessels from a small boat while both are moving.",
                "The depth shown on a chart is measured from the lowest predicted tide."]),
    dict(name="bees",
         actors=["the beekeeper", "a volunteer from the allotment society", "the apiary inspector",
                 "two apprentices", "the honey packer", "a retired schoolteacher", "the club secretary",
                 "the regional bee officer"],
         objects=["the brood frames", "a new queen", "the smoker", "the top bar hive", "the wax foundation",
                  "the honey extractor", "the entrance reducer", "the swarm trap", "the feeding syrup",
                  "the mite counts"],
         verbs=_v("check/checks/checked/checking", "move/moves/moved/moving", "label/labels/labelled/labelling",
                  "treat/treats/treated/treating", "weigh/weighs/weighed/weighing",
                  "replace/replaces/replaced/replacing"),
         places=["the south orchard", "the heather moor", "the shed behind the school", "the river meadow",
                 "the roof garden", "the training apiary"],
         facts=["A worker bee visits several hundred flowers on a single foraging trip.",
                "Colonies cluster around the queen in winter and shiver to keep her warm.",
                "Varroa mites are counted by placing a sticky board under the hive for three days.",
                "Honey is ripe when the bees have capped most of the cells with wax.",
                "Swarming is most likely on warm afternoons in late spring.",
                "A queen can lay more than a thousand eggs on a good day."]),
    dict(name="mill",
         actors=["the mill manager", "a group of weavers", "the dye master", "the museum curator",
                 "an apprentice mechanic", "the loom fitter", "a local historian", "the night watchman"],
         objects=["the power loom", "the steam engine", "the spinning frames", "the dye vats",
                  "the ledger of wages", "the weaving shed roof", "the water wheel", "the bobbin store",
                  "the boiler house", "the pattern cards"],
         verbs=_v("restore/restores/restored/restoring", "oil/oils/oiled/oiling", "catalogue/catalogues/catalogued/cataloguing",
                  "dismantle/dismantles/dismantled/dismantling", "photograph/photographs/photographed/photographing",
                  "demonstrate/demonstrates/demonstrated/demonstrating"),
         places=["the upper floor", "the canal side", "the engine hall", "the finishing room",
                 "the old counting house", "the weaving shed"],
         facts=["Jacquard cards store a woven pattern as rows of punched holes.",
                "Water power was cheap but unreliable in a dry summer.",
                "Cotton fibres break more easily when the air in the shed is too dry.",
                "A single weaver once tended two or three looms at the same time.",
                "Mill chimneys were built tall to pull air through the boilers.",
                "Wages were often recorded in a ledger kept by the overseer."]),
    dict(name="satellite",
         actors=["the flight controller", "a ground station operator", "the payload scientist",
                 "the thermal engineer", "a graduate student", "the mission planner", "the launch coordinator",
                 "the antenna technician"],
         objects=["the reaction wheel", "the star tracker", "the solar array", "the downlink schedule",
                  "the onboard computer", "the S-band antenna", "the battery pack", "the thruster valve",
                  "the camera payload", "the telemetry archive"],
         verbs=_v("upload/uploads/uploaded/uploading", "simulate/simulates/simulated/simulating",
                  "diagnose/diagnoses/diagnosed/diagnosing", "reboot/reboots/rebooted/rebooting",
                  "calibrate/calibrates/calibrated/calibrating", "monitor/monitors/monitored/monitoring"),
         places=["the control room", "the clean room", "the rooftop dish", "the test chamber",
                 "the backup ground station", "the integration bay"],
         facts=["A small satellite in low orbit circles the Earth roughly every ninety minutes.",
                "Ground stations can only talk to a satellite while it is above the horizon.",
                "Batteries age faster when they are charged to the top on every orbit.",
                "A star tracker finds its orientation by matching patterns of stars to a catalogue.",
                "Radiation can flip a single bit in memory, so critical data is stored with checks.",
                "The side facing the sun can be a hundred degrees hotter than the side in shadow."]),
    dict(name="cycling",
         actors=["the city transport planner", "a neighbourhood cycling group", "the road surface crew",
                 "a local councillor", "the traffic analyst", "two volunteers with counters",
                 "the school travel officer", "the bike shop owner"],
         objects=["the protected lane", "the junction signals", "the bike parking racks", "the resurfacing plan",
                  "the bridge ramp", "the counter data", "the speed humps", "the lighting upgrade",
                  "the shared path", "the pilot scheme"],
         verbs=_v("survey/surveys/surveyed/surveying", "widen/widens/widened/widening", "approve/approves/approved/approving",
                  "paint/paints/painted/painting", "count/counts/counted/counting",
                  "redesign/redesigns/redesigned/redesigning"),
         places=["the high street", "the riverside route", "the railway underpass", "the market square",
                 "the ring road", "the hospital approach"],
         facts=["Painted lanes help less than physically separated ones on busy roads.",
                "Most cycling injuries happen at junctions, not on straight stretches.",
                "Riders choose a slightly longer route if it feels calmer.",
                "Secure parking at both ends of a trip matters as much as the lane itself.",
                "Counters buried in the road surface record every wheel that passes.",
                "Winter gritting of cycle paths is often forgotten in the maintenance budget."]),
    dict(name="bakery",
         actors=["the head baker", "a pastry apprentice", "the bakery owner", "the delivery driver",
                 "the night crew", "a food safety inspector", "the sourdough specialist", "the café manager"],
         objects=["the sourdough starter", "the deck oven", "the proofing cabinet", "the rye dough",
                  "the croissant lamination", "the flour order", "the cooling racks", "the dough mixer",
                  "the glaze", "the morning bake list"],
         verbs=_v("shape/shapes/shaped/shaping", "fold/folds/folded/folding", "proof/proofs/proofed/proofing",
                  "score/scores/scored/scoring", "weigh/weighs/weighed/weighing",
                  "refresh/refreshes/refreshed/refreshing"),
         places=["the front counter", "the cold room", "the back kitchen", "the market stall",
                 "the loading door", "the prep bench"],
         facts=["A sourdough starter is a living mix of wild yeast and bacteria that is fed with flour and water.",
                "Steam in the first minutes of baking keeps the crust flexible so the loaf can rise.",
                "Cold dough is easier to shape but takes longer to proof.",
                "Butter layers in a croissant melt in the oven and puff the dough apart.",
                "Flour absorbs different amounts of water depending on the season.",
                "A loaf is done when it sounds hollow if tapped on the base."]),
    dict(name="glacier",
         actors=["the field glaciologist", "a survey team", "the hydrology student", "the station manager",
                 "the helicopter pilot", "a drone operator", "the data analyst", "the mountain guide"],
         objects=["the ablation stakes", "the ice core", "the weather mast", "the meltwater channel",
                  "the terminus marker", "the snow pit", "the GPS beacon", "the radar profile",
                  "the sediment trap", "the camera time-lapse"],
         verbs=_v("drill/drills/drilled/drilling", "measure/measures/measured/measuring",
                  "map/maps/mapped/mapping", "retrieve/retrieves/retrieved/retrieving",
                  "replace/replaces/replaced/replacing", "compare/compares/compared/comparing"),
         places=["the lower tongue", "the accumulation basin", "the base camp", "the moraine ridge",
                 "the ice cave", "the western icefall"],
         facts=["Stakes drilled into the ice show how many metres of surface melt away each summer.",
                "Layers in an ice core record snowfall, dust and trapped air from past years.",
                "Meltwater often drains through tunnels inside the glacier before reaching the valley.",
                "A glacier can keep moving forward even while its front edge retreats.",
                "Dark dust on the surface makes ice absorb more sunlight and melt faster.",
                "Radar can see the bed of a glacier through hundreds of metres of ice."]),
    dict(name="library",
         actors=["the head librarian", "a cataloguing assistant", "the archive volunteer", "the reading room clerk",
                 "the conservation officer", "a visiting scholar", "the systems librarian", "the children's room lead"],
         objects=["the card catalogue", "the rare map collection", "the digitisation queue", "the loan records",
                  "the shelving plan", "the donated letters", "the periodicals index", "the humidity logger",
                  "the reading lamps", "the interlibrary requests"],
         verbs=_v("classify/classifies/classified/classifying", "scan/scans/scanned/scanning",
                  "shelve/shelves/shelved/shelving", "repair/repairs/repaired/repairing",
                  "index/indexes/indexed/indexing", "withdraw/withdraws/withdrew/withdrawing"),
         places=["the basement stacks", "the reading room", "the loading bay", "the map room",
                 "the local history section", "the conservation studio"],
         facts=["Acid in cheap paper slowly turns pages brown and brittle.",
                "A classification scheme puts books on similar subjects next to each other.",
                "Rare maps are stored flat because folding cracks the paper.",
                "Scanning a fragile book can take longer than reading it.",
                "Metadata, not the scanned image, is what lets people find an item later.",
                "Light and humidity damage archives faster than handling does."]),
    dict(name="wind",
         actors=["the turbine technician", "the site manager", "a rope access team", "the grid operator",
                 "the blade inspector", "the control room analyst", "a boat captain", "the landowner liaison"],
         objects=["the gearbox", "the pitch system", "the blade root bolts", "the nacelle cooling fan",
                  "the cable joint", "the yaw bearing", "the anemometer", "the transformer",
                  "the access ladder", "the vibration sensor"],
         verbs=_v("inspect/inspects/inspected/inspecting", "torque/torques/torqued/torquing",
                  "replace/replaces/replaced/replacing", "lubricate/lubricates/lubricated/lubricating",
                  "log/logs/logged/logging", "shut down/shuts down/shut down/shutting down"),
         places=["turbine fourteen", "the offshore platform", "the substation", "the service port",
                 "the northern ridge", "the maintenance depot"],
         facts=["A turbine turns itself to face the wind using a slow motor under the nacelle.",
                "Output grows with the cube of wind speed until the turbine reaches its rated power.",
                "Gearbox oil is sampled and tested for metal particles that warn of wear.",
                "Blades are shut down and feathered when the wind becomes dangerously strong.",
                "Technicians wait for a calm window before climbing offshore towers.",
                "Cables to the shore carry power at a much higher voltage than the turbines make."]),
    dict(name="theatre",
         actors=["the director", "a community volunteer", "the stage manager", "the lighting designer",
                 "the lead actor", "the costume maker", "the box office team", "the sound operator"],
         objects=["the opening scene", "the prompt book", "the revolving set", "the lighting cues",
                  "the period costumes", "the ticket sales", "the dress rehearsal", "the programme booklet",
                  "the stage floor", "the final act"],
         verbs=_v("rehearse/rehearses/rehearsed/rehearsing", "rewrite/rewrites/rewrote/rewriting",
                  "cue/cues/cued/cueing", "stitch/stitches/stitched/stitching",
                  "advertise/advertises/advertised/advertising", "block/blocks/blocked/blocking"),
         places=["the village hall", "the green room", "the back of the stalls", "the wings",
                 "the rehearsal studio", "the foyer"],
         facts=["Blocking is the plan for where each actor stands and moves during a scene.",
                "A prompt book holds the script, every cue and every note for the show.",
                "Stage lights are focused one by one before the first rehearsal with the cast.",
                "Community productions often borrow props and costumes from neighbours.",
                "A dress rehearsal is run as if an audience were watching and without stopping.",
                "Sound cues are timed to the second so effects land with the action."]),
    dict(name="vineyard",
         actors=["the vineyard manager", "a team of pickers", "the winemaker", "the cellar hand",
                 "a visiting agronomist", "the tractor driver", "the sommelier", "the estate owner"],
         objects=["the trellis wires", "the Riesling block", "the press", "the fermentation tanks",
                  "the frost candles", "the barrel room", "the soil samples", "the irrigation line",
                  "the harvest crates", "the bottling line"],
         verbs=_v("prune/prunes/pruned/pruning", "taste/tastes/tasted/tasting", "pick/picks/picked/picking",
                  "rack/racks/racked/racking", "spray/sprays/sprayed/spraying", "sample/samples/sampled/sampling"),
         places=["the lower slope", "the south-facing terrace", "the cellar", "the tasting room",
                 "the old stone press house", "the valley floor"],
         facts=["Grapes are picked when the balance of sugar and acid is right, not by the calendar.",
                "Late spring frost can destroy a year's crop in a single night.",
                "White wine is fermented without the grape skins, red wine with them.",
                "Barrels give a wine small amounts of oxygen and flavour as it ages.",
                "Vines on poor, stony soil often make more concentrated fruit.",
                "Sulphur is added in tiny amounts to protect wine from spoiling."]),
    dict(name="railway",
         actors=["the signaller", "a track maintenance gang", "the station master", "the train planner",
                 "the driver trainer", "the level crossing inspector", "a heritage volunteer", "the depot engineer"],
         objects=["the points motor", "the timetable revision", "the signal lamp", "the overhead wire",
                  "the ballast", "the platform extension", "the brake test", "the axle counter",
                  "the turntable", "the freight siding"],
         verbs=_v("renew/renews/renewed/renewing", "test/tests/tested/testing", "reroute/reroutes/rerouted/rerouting",
                  "tamp/tamps/tamped/tamping", "announce/announces/announced/announcing",
                  "isolate/isolates/isolated/isolating")
         , places=["the junction", "the signal box", "the carriage sheds", "the viaduct",
                   "the terminus", "the passing loop"],
         facts=["Track is divided into blocks so that only one train is in each section at a time.",
                "Rails expand in summer heat, so gaps or special fastenings are needed.",
                "Ballast stones under the sleepers spread the load and let water drain.",
                "A signal shows danger by default and clears only when the line ahead is free.",
                "Timetables leave small gaps so one late train does not delay every other.",
                "Steam locomotives took on water every hundred kilometres or so."]),
    dict(name="clinic",
         actors=["the practice nurse", "a locum doctor", "the clinic receptionist", "the pharmacy technician",
                 "the practice manager", "two physiotherapists", "the community midwife", "a medical student"],
         objects=["the appointment diary", "the vaccine fridge", "the blood pressure cuffs", "the waiting room",
                  "the referral letters", "the stock of dressings", "the patient leaflets", "the sterilising unit",
                  "the walk-in rota", "the test results inbox"],
         verbs=_v("review/reviews/reviewed/reviewing", "restock/restocks/restocked/restocking",
                  "book/books/booked/booking", "sterilise/sterilises/sterilised/sterilising",
                  "update/updates/updated/updating", "explain/explains/explained/explaining"),
         places=["the treatment room", "the front desk", "the dispensary", "the annexe",
                 "the child health clinic", "the car park"],
         facts=["Vaccines must stay within a narrow temperature range from the factory to the arm.",
                "A missed appointment costs the clinic more than the empty slot suggests.",
                "Blood pressure readings vary with the size of the cuff and how long the patient has sat.",
                "Plain language leaflets are read more often than long ones.",
                "Instruments are sterilised in sealed pouches that show when the cycle has run.",
                "Referral letters are tracked so none is lost between services."]),
    dict(name="farm",
         actors=["the dairy farmer", "a contract milker", "the vet", "the feed supplier",
                 "the farm accountant", "the apprentice shepherd", "the hedge layer", "the neighbouring grower"],
         objects=["the milking parlour", "the silage clamp", "the lambing pens", "the slurry store",
                  "the hay baler", "the gate hinges", "the cow tags", "the bulk tank",
                  "the field drains", "the tractor tyres"],
         verbs=_v("clean/cleans/cleaned/cleaning", "inspect/inspects/inspected/inspecting",
                  "cut/cuts/cut/cutting", "move/moves/moved/moving", "treat/treats/treated/treating",
                  "record/records/recorded/recording"),
         places=["the top field", "the yard", "the lower pasture", "the barn", "the lane end", "the dairy"],
         facts=["Grass cut at the right moment makes silage that cows eat eagerly through winter.",
                "A cow needs many litres of water for every litre of milk she makes.",
                "Hedges laid by hand stay thick at the bottom and keep sheep in.",
                "Lambing is busiest at night, so the pens are checked every few hours.",
                "Field drains clog with roots and need rodding every few years.",
                "Milk is cooled within hours so bacteria cannot grow."]),
    dict(name="observatory",
         actors=["the night astronomer", "an amateur observer", "the telescope technician", "the outreach officer",
                 "a doctoral student", "the dome operator", "the instrument scientist", "the volunteer guide"],
         objects=["the primary mirror", "the guide camera", "the filter wheel", "the observing log",
                  "the dome motor", "the spectrograph slit", "the weather station", "the flat-field frames",
                  "the open night schedule", "the cooling system"],
         verbs=_v("align/aligns/aligned/aligning", "clean/cleans/cleaned/cleaning", "point/points/pointed/pointing",
                  "stack/stacks/stacked/stacking", "book/books/booked/booking",
                  "analyse/analyses/analysed/analysing"),
         places=["the main dome", "the warm room", "the visitor deck", "the mountain road",
                 "the roll-off roof shed", "the data office"],
         facts=["Cameras on telescopes are cooled so that heat does not add noise to faint images.",
                "Stacking many short exposures reveals objects too faint for any single frame.",
                "The sky is darkest on moonless nights far from city lights.",
                "Mirrors are re-coated with a thin layer of aluminium every few years.",
                "Flat-field frames correct for dust specks and uneven sensitivity in the detector.",
                "Turbulent air makes stars twinkle and blurs long exposures."]),
    dict(name="school",
         actors=["the head of year", "a maths teacher", "the school caretaker", "the parent association chair",
                 "the librarian", "a class of nine-year-olds", "the exams officer", "the music teacher"],
         objects=["the timetable", "the science lab", "the sports day", "the reading scheme",
                  "the school garden", "the end-of-term concert", "the homework policy", "the lunch menu",
                  "the minibus", "the fire drill"],
         verbs=_v("plan/plans/planned/planning", "supervise/supervises/supervised/supervising",
                  "mark/marks/marked/marking", "organise/organises/organised/organising",
                  "revise/revises/revised/revising", "announce/announces/announced/announcing"),
         places=["the main hall", "the playground", "classroom 4B", "the staff room",
                 "the library corner", "the music room"],
         facts=["Short, frequent practice helps children remember number facts better than a long session.",
                "A fire drill is timed so that staff know how long the building takes to empty.",
                "Reading aloud together builds vocabulary even for confident readers.",
                "School gardens turn lessons about plants into jobs that need doing every week.",
                "A timetable is a puzzle: rooms, teachers and groups all have to fit at once.",
                "Concert rehearsals are easier when each class learns its own part first."]),
]

QUALS = ["careful", "unusually thorough", "hurried", "well-documented", "cautious", "patient", "ambitious",
         "routine", "carefully timed", "improvised", "long overdue", "surprisingly cheap", "stubborn",
         "late", "tidy", "slow but steady", "last-minute", "meticulous"]
PERIODS = ["morning", "week", "month", "quarter", "season", "Friday", "evening", "year"]
SEASONS = ["spring", "summer", "autumn", "winter"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
OUTCOMES = ["delays fell by a third", "complaints dropped noticeably", "costs stayed within the budget",
            "nobody noticed the difference at first", "the schedule slipped by two days",
            "the backlog finally cleared", "the results were easier to compare", "the team gained a day each week",
            "a small fault was caught early", "the old method was quietly retired",
            "the numbers matched the earlier estimate", "several people asked for the same change"]
REASONS = ["the earlier readings did not agree", "a colleague had raised a concern", "the old approach kept failing",
           "the weather forced a change of plan", "there was a gap in the records", "a supplier delivered late",
           "the budget had been cut", "an audit was coming", "the equipment was due for a service",
           "new rules had just come into force"]
CONNECT = ["In practice,", "However,", "At the same time,", "As a result,", "Even so,", "For that reason,",
           "Meanwhile,", "Later on,", "By contrast,", "In most cases,", "Still,", "To be fair,"]

FIRST = ["Maren", "Idris", "Collette", "Tomasz", "Ngozi", "Halvard", "Priya", "Joaquin", "Linnea", "Okonkwo",
         "Saoirse", "Dmitri", "Amara", "Bastien", "Yusuf", "Elodie", "Anselm", "Keiko", "Rafael", "Thandi",
         "Matteo", "Ingrid", "Farid", "Odette", "Lukas", "Zainab", "Casimir", "Noor", "Henrik", "Marisol",
         "Emeka", "Greta", "Jonas", "Soledad", "Tariq", "Winifred", "Anouk", "Bartek", "Chiara", "Devraj"]
LAST = ["Aldous", "Brennan", "Castellano", "Dvorak", "Eriksen", "Fairweather", "Gupta", "Haldane", "Ibarra",
        "Jovanovic", "Kowalczyk", "Lindqvist", "Moreau", "Nakamura", "Oyelaran", "Pemberton", "Quint", "Rasmussen",
        "Sandoval", "Tennant", "Umarov", "Vasquez", "Whitlock", "Xavier", "Yeung", "Zielinski", "Abernathy",
        "Boucher", "Cardoso", "Delacroix", "Ekwueme", "Fontaine", "Grimaldi", "Hartmann", "Iyer", "Janssen",
        "Kaplan", "Lombardi", "Mbeki", "Novak"]
TOWNS = ["Aldermoor", "Brightwater", "Carrowbeg", "Dunmere", "Eastholt", "Fenwick Cross", "Glenavon", "Hartlow",
         "Ironbridge Quay", "Jarrowgate", "Kilbrae", "Lowmoor", "Marlbury", "Northcote", "Oakhaven", "Pennard",
         "Quarry Bank", "Ravensford", "Stonemarsh", "Thornley"]
PRODUCTS = ["Kestrel", "Lumen", "Borealis", "Tessera", "Quillon", "Marigold", "Corvid", "Halcyon", "Ember", "Juniper",
            "Obsidian", "Pennant", "Sable", "Trellis", "Vesper", "Willow"]
THINGS = ["lantern", "ledger", "compass", "kettle", "satchel", "barometer", "thermos", "toolbox", "radio",
          "notebook", "umbrella", "backpack", "tripod", "headset", "scanner", "printer"]
WORDS = ["amber", "birch", "cobalt", "dune", "ember", "fjord", "garnet", "harbor", "indigo", "juniper", "kelp",
         "lotus", "mesa", "nectar", "onyx", "pine", "quartz", "reef", "sage", "tundra", "umber", "velvet",
         "willow", "yarrow", "zephyr"]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


class Writer:
    """A seeded text source drawing only on the given domain indices."""

    def __init__(self, rng: random.Random, domains: list[int],
                 names: tuple[int, int] = (0, 1)) -> None:
        self.rng = rng
        self.domains = [DOMAINS[i] for i in domains]
        self.first = FIRST[names[0]::2]
        self.last = LAST[names[0]::2]
        self.towns = TOWNS[names[0]::2]
        self.products = PRODUCTS[names[0]::2]
        self.dom = self.domains[0]

    # ---- small pieces -----------------------------------------------------
    def pick(self, seq):
        return seq[self.rng.randrange(len(seq))]

    def name(self) -> str:
        return f"{self.pick(self.first)} {self.pick(self.last)}"

    def switch_domain(self) -> None:
        self.dom = self.pick(self.domains)

    def _slots(self) -> dict:
        d, r = self.dom, self.rng
        v = self.pick(d["verbs"])
        v2 = self.pick(d["verbs"])
        q = self.pick(QUALS)
        return dict(A=self.pick(d["actors"]), A2=self.pick(d["actors"]), O=self.pick(d["objects"]),
                    O2=self.pick(d["objects"]), P=self.pick(d["places"]), P2=self.pick(d["places"]),
                    v=v[0], v3=v[1], vp=v[2], vi=v[3], w3=v2[1], wp=v2[2], wi=v2[3],
                    Q=q, aQ=("an " if q[0] in "aeiou" else "a ") + q, R=self.pick(REASONS), X=self.pick(OUTCOMES),
                    T=self.pick(PERIODS), S=self.pick(SEASONS), N=r.randrange(3, 97),
                    N2=r.randrange(2, 14), Y=r.randrange(1962, 2025), M=self.pick(MONTHS),
                    NAME=self.name(), F=self.pick(d["facts"]))

    _TEMPLATES = [
        "{A} {v3} {O} at {P} every {T}.",
        "Last {S}, {A} {vp} {O} after {R}, and {X}.",
        "{A} {vp} {O} in {M} {Y}, {aQ} job that took {N} days.",
        "Because {R}, {A} {w3} {O} twice as often as before.",
        "{F}",
        "{F} That is why {A} {v3} {O} with such care.",
        "{A} spent {N2} hours {wi} {O} at {P}, then {vp} {O2} as well.",
        "According to {NAME}, {O} had never been {Q} until {A} {vp} it.",
        "Before {A} {w3} {O}, {A2} {vp} {O2} at {P2}; afterwards {X}.",
        "It was {NAME} who first suggested {vi} {O}, and {X}.",
        "Nobody at {P} expected {A} to {v} {O} so early, yet {X}.",
        "Over {N2} weeks, {A} {vp} {O} and {wp} {O2}, {aQ} routine, and {X}.",
        "{O} sits near {P}, where {A} {v3} it each {T}.",
        "When {R}, {A} had to {v} {O} by hand.",
        "The {Q} approach {A} took with {O} meant that {X}.",
        "By {M} {Y}, {A} had {vp} {O} more than {N} times.",
        "{A2} asked whether {O2} could be {vp} at {P2} instead, but {A} said it would not work.",
        "In a note from {M}, {NAME} wrote that {O} should be {vp} before the next {S}.",
        "{O} and {O2} were both moved to {P} so that {A} could {v} them together.",
        "Few people knew that {A} {v3} {O} on {N2} separate occasions that {S}.",
    ]

    def sentence(self) -> str:
        t = self.pick(self._TEMPLATES)
        s = t.format(**self._slots())
        return _cap(s)

    def paragraph(self, n_sentences: int | None = None) -> str:
        self.switch_domain()
        n = n_sentences or self.rng.randrange(4, 8)
        out = []
        for i in range(n):
            s = self.sentence()
            if i and self.rng.random() < 0.35:
                s = f"{self.pick(CONNECT)} {s[:1].lower()}{s[1:]}"
            out.append(s)
        return " ".join(out)

    def title(self) -> str:
        sl = self._slots()
        return self.pick([
            "Notes on {O} at {P}", "How {A} {v3} {O}", "{aQ} year for {O}", "Why {O} matters",
            "Field report: {P}", "Lessons from {N2} seasons of {wi} {O}",
            "What changed after {M} {Y}"]).format(**sl).replace("the ", "", 1) \
            .replace("The ", "", 1).title()

    # ---- documents ----------------------------------------------------------
    def _article(self) -> str:
        self.switch_domain()
        parts = [f"# {self.title()}", ""]
        for _ in range(self.rng.randrange(4, 8)):
            if self.rng.random() < 0.6:
                parts += [f"## {self.title()}", ""]
            parts += [self.paragraph(), ""]
        return "\n".join(parts)

    def _minutes(self) -> str:
        self.switch_domain()
        sl = self._slots()
        lines = [f"Minutes of the {self.pick(PERIODS)} meeting, {sl['P']}, {sl['M']} {sl['Y']}",
                 "Present: " + ", ".join(self.name() for _ in range(self.rng.randrange(3, 7))), ""]
        for i in range(self.rng.randrange(3, 7)):
            lines.append(f"{i + 1}. {self.title()}")
            lines.append("   " + self.paragraph(self.rng.randrange(2, 4)))
        lines.append("")
        lines.append("Actions:")
        for _ in range(self.rng.randrange(3, 6)):
            s = self._slots()
            lines.append(f"- {s['NAME']} to {s['v']} {s['O']} by {s['M']} {s['N2']}.")
        return "\n".join(lines) + "\n"

    def _email_thread(self) -> str:
        self.switch_domain()
        people = [self.name() for _ in range(3)]
        out = [f"Subject: {self.title()}", ""]
        for i in range(self.rng.randrange(3, 6)):
            a, b = people[i % 2], people[(i + 1) % 2]
            out += [f"From: {a}", f"To: {b}", f"Date: {self.pick(MONTHS)} {self.rng.randrange(1, 29)}", "",
                    f"Hi {b.split()[0]},", "", self.paragraph(self.rng.randrange(2, 5)), "",
                    f"Thanks,\n{a.split()[0]}", "", ">" * 3, ""]
        return "\n".join(out)

    def _faq(self) -> str:
        self.switch_domain()
        out = [f"Frequently asked questions: {self.title()}", ""]
        for _ in range(self.rng.randrange(4, 8)):
            s = self._slots()
            q = self.pick(["Who is responsible for {O}?", "When does {A} {v} {O}?", "What happens if {R}?",
                           "Why was {O} moved to {P}?", "How often is {O} checked?"]).format(**s)
            out += [f"Q: {q}", f"A: {self.paragraph(self.rng.randrange(2, 4))}", ""]
        return "\n".join(out)

    def _story(self) -> str:
        self.switch_domain()
        out = []
        for _ in range(self.rng.randrange(5, 9)):
            s = self._slots()
            out.append(self.paragraph(self.rng.randrange(3, 5)))
            out.append(f"\"{_cap(self.pick(REASONS))},\" said {s['NAME']}. \"{_cap(s['A'])} will {s['v']} {s['O']} "
                       f"at {s['P']} on {s['M']} {s['N2']}.\"")
        return "\n\n".join(out) + "\n"

    def _report(self) -> str:
        self.switch_domain()
        out = [f"Report: {self.title()}", "", self.paragraph(), "",
               "| item | count | cost | status |", "|---|---|---|---|"]
        for _ in range(self.rng.randrange(5, 10)):
            s = self._slots()
            out.append(f"| {s['O'].replace('the ', '', 1)} | {s['N']} | {self.rng.randrange(10, 9000)} | "
                       f"{self.pick(['done', 'open', 'blocked', 'in review'])} |")
        out += ["", self.paragraph(), ""]
        return "\n".join(out)

    def document(self, min_chars: int) -> str:
        kinds = [self._article, self._article, self._minutes, self._email_thread, self._faq, self._story,
                 self._report]
        chunks, size = [], 0
        while size < min_chars:
            c = self.pick(kinds)()
            chunks.append(c)
            size += len(c) + 8
        return "\n\n---\n\n".join(chunks)

    # ---- code -------------------------------------------------------------------
    def _ident(self, kind: str = "var") -> str:
        d = self.dom
        base = re.sub(r"[^a-z]+", "_", self.pick(d["objects"]).replace("the ", "", 1).replace("a ", "", 1)).strip("_")
        return base if kind == "var" else _cap(base.replace("_", " ")).replace(" ", "")

    def _py_func(self) -> tuple[str, str]:
        r = self.rng
        name = f"{self.pick(['compute', 'collect', 'merge', 'filter', 'summarise', 'resolve', 'rank', 'scan', 'build', 'check'])}_{self._ident()}"
        a, b = self._ident(), self._ident()
        if a == b:
            b += "_2"
        k = r.randrange(2, 40)
        shapes = [
            (f"def {name}(items, limit={k}):\n    \"\"\"Return the first `limit` items whose weight is above the mean.\"\"\"\n"
             f"    if not items:\n        return []\n    mean = sum(i['weight'] for i in items) / len(items)\n"
             f"    out = [i for i in items if i['weight'] > mean]\n    out.sort(key=lambda i: (-i['weight'], i['id']))\n"
             f"    return out[:limit]\n"),
            (f"def {name}(rows, key='{a}'):\n    \"\"\"Group rows by `key` and count them.\"\"\"\n    counts = {{}}\n"
             f"    for row in rows:\n        counts[row[key]] = counts.get(row[key], 0) + 1\n"
             f"    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))))\n"),
            (f"def {name}(values, window={r.randrange(2, 9)}):\n    \"\"\"Moving average with a fixed window; shorter prefixes use what exists.\"\"\"\n"
             f"    out, total = [], 0.0\n    for i, v in enumerate(values):\n        total += v\n"
             f"        if i >= window:\n            total -= values[i - window]\n"
             f"        out.append(total / min(i + 1, window))\n    return out\n"),
            (f"def {name}(sorted_{a}, target):\n    \"\"\"Binary search; return the index of target or -1.\"\"\"\n"
             f"    lo, hi = 0, len(sorted_{a}) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n"
             f"        if sorted_{a}[mid] == target:\n            return mid\n"
             f"        if sorted_{a}[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n"),
            (f"def {name}(intervals):\n    \"\"\"Merge overlapping (start, end) pairs.\"\"\"\n    merged = []\n"
             f"    for start, end in sorted(intervals):\n        if merged and start <= merged[-1][1]:\n"
             f"            merged[-1] = (merged[-1][0], max(merged[-1][1], end))\n        else:\n"
             f"            merged.append((start, end))\n    return merged\n"),
            (f"def {name}(text, width={r.randrange(30, 90)}):\n    \"\"\"Wrap text at word boundaries.\"\"\"\n    lines, line = [], ''\n"
             f"    for word in text.split():\n        if len(line) + len(word) + 1 > width and line:\n"
             f"            lines.append(line)\n            line = word\n        else:\n"
             f"            line = f'{{line}} {{word}}'.strip()\n    if line:\n        lines.append(line)\n    return lines\n"),
            (f"def {name}(graph):\n    \"\"\"Topological order of a dict of node -> list of dependencies.\"\"\"\n"
             f"    order, seen, active = [], set(), set()\n    def visit(node):\n        if node in seen:\n            return\n"
             f"        if node in active:\n            raise ValueError(f'cycle at {{node}}')\n        active.add(node)\n"
             f"        for dep in graph.get(node, []):\n            visit(dep)\n        active.discard(node)\n"
             f"        seen.add(node)\n        order.append(node)\n    for node in sorted(graph):\n        visit(node)\n    return order\n"),
            (f"def {name}(line, sep=','):\n    \"\"\"Split a CSV line, honouring double quotes.\"\"\"\n    fields, cur, quoted = [], [], False\n"
             f"    for ch in line:\n        if ch == '\"':\n            quoted = not quoted\n        elif ch == sep and not quoted:\n"
             f"            fields.append(''.join(cur))\n            cur = []\n        else:\n            cur.append(ch)\n"
             f"    fields.append(''.join(cur))\n    return fields\n"),
            (f"def {name}(a, b):\n    \"\"\"Edit distance between two strings.\"\"\"\n    prev = list(range(len(b) + 1))\n"
             f"    for i, ca in enumerate(a, 1):\n        cur = [i]\n        for j, cb in enumerate(b, 1):\n"
             f"            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))\n        prev = cur\n"
             f"    return prev[-1]\n"),
            (f"def {name}(n, memo={{}}):\n    \"\"\"Number of ways to climb n steps taking 1, 2 or {r.randrange(3, 5)} at a time.\"\"\"\n"
             f"    if n < 0:\n        return 0\n    if n == 0:\n        return 1\n    if n not in memo:\n"
             f"        memo[n] = {name}(n - 1) + {name}(n - 2) + {name}(n - 3)\n    return memo[n]\n"),
            (f"def {name}(path, retries={r.randrange(2, 6)}):\n    \"\"\"Read a file, retrying on transient errors.\"\"\"\n    import time\n"
             f"    last = None\n    for attempt in range(retries):\n        try:\n            with open(path, encoding='utf-8') as fh:\n"
             f"                return fh.read()\n        except OSError as exc:\n            last = exc\n"
             f"            time.sleep(0.05 * 2 ** attempt)\n    raise last\n"),
            (f"def {name}(samples, q=0.{r.randrange(5, 10)}):\n    \"\"\"Quantile by linear interpolation.\"\"\"\n    xs = sorted(samples)\n"
             f"    if not xs:\n        raise ValueError('no samples')\n    pos = q * (len(xs) - 1)\n    lo = int(pos)\n"
             f"    hi = min(lo + 1, len(xs) - 1)\n    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)\n"),
        ]
        src = self.pick(shapes)
        return name, src

    def python_module(self, n_funcs: int) -> tuple[str, list[str]]:
        self.switch_domain()
        names, bodies = [], []
        while len(bodies) < n_funcs:
            name, src = self._py_func()
            if name in names:
                continue
            names.append(name)
            bodies.append(src)
        head = ['"""' + _cap(self._slots()['O'].replace('the ', '', 1)) + ' helpers for ' +
                self.pick(self.towns) + '."""', "from __future__ import annotations", "", ""]
        return "\n\n".join(["\n".join(head).rstrip() + "\n"] + bodies), names

    def python_completion(self, min_chars: int) -> str:
        """A Python file that stops after the signature and docstring of a
        function it has not written yet."""
        self.switch_domain()
        names: list[str] = []
        parts = ['"""' + _cap(self._slots()['O'].replace('the ', '', 1)) + ' helpers for ' +
                 self.pick(self.towns) + '."""\nfrom __future__ import annotations\n']
        size = len(parts[0])
        while size < min_chars:
            name, src = self._py_func()
            if name not in names:
                names.append(name)
                parts.append(src)
                size += len(src) + 2
        while True:
            name, src = self._py_func()
            if name not in names:
                return "\n\n".join(parts) + "\n\n" + "\n".join(src.split("\n")[:2]) + "\n"

    def _py_class(self) -> str:
        cls = self._ident("cls") + self.pick(["Store", "Planner", "Tracker", "Registry", "Queue", "Index"])
        a = self._ident()
        k = self.rng.randrange(3, 30)
        return (f"class {cls}:\n    \"\"\"Keeps a bounded list of {a} records.\"\"\"\n\n"
                f"    def __init__(self, capacity={k}):\n        self.capacity = capacity\n        self._rows = []\n\n"
                f"    def add(self, row):\n        self._rows.append(row)\n        if len(self._rows) > self.capacity:\n"
                f"            self._rows.pop(0)\n        return len(self._rows)\n\n"
                f"    def find(self, **where):\n        return [r for r in self._rows\n"
                f"                if all(r.get(k) == v for k, v in where.items())]\n\n"
                f"    def __len__(self):\n        return len(self._rows)\n")

    def _js(self) -> str:
        a, b = self._ident(), self._ident()
        n = self.rng.randrange(2, 20)
        return self.pick([
            f"export function debounce_{a}(fn, ms = {n * 10}) {{\n  let timer = null;\n  return (...args) => {{\n"
            f"    clearTimeout(timer);\n    timer = setTimeout(() => fn(...args), ms);\n  }};\n}}\n",
            f"export async function fetch_{a}(url, retries = {n % 5 + 1}) {{\n  for (let i = 0; i < retries; i++) {{\n"
            f"    const res = await fetch(url);\n    if (res.ok) return res.json();\n"
            f"    await new Promise((r) => setTimeout(r, 100 * 2 ** i));\n  }}\n  throw new Error(`failed: ${{url}}`);\n}}\n",
            f"export const group_{a} = (rows, key) =>\n  rows.reduce((acc, row) => {{\n    (acc[row[key]] ||= []).push(row);\n"
            f"    return acc;\n  }}, {{}});\n",
            f"export function clamp_{a}(x, lo = 0, hi = {n * 5}) {{\n  return Math.min(hi, Math.max(lo, x));\n}}\n",
            f"class {self._ident('cls')}Cache {{\n  constructor(limit = {n}) {{\n    this.limit = limit;\n    this.map = new Map();\n  }}\n"
            f"  get(key) {{\n    if (!this.map.has(key)) return undefined;\n    const v = this.map.get(key);\n"
            f"    this.map.delete(key);\n    this.map.set(key, v);\n    return v;\n  }}\n"
            f"  set(key, v) {{\n    this.map.delete(key);\n    this.map.set(key, v);\n"
            f"    if (this.map.size > this.limit) this.map.delete(this.map.keys().next().value);\n  }}\n}}\n",
        ])

    def _sql(self) -> str:
        t, u = self._ident(), self._ident()
        n = self.rng.randrange(2, 90)
        return self.pick([
            f"SELECT {t}_id, COUNT(*) AS n, AVG(cost) AS avg_cost\nFROM {t}_log\nWHERE created_at >= DATE '2024-0{n % 9 + 1}-01'\n"
            f"GROUP BY {t}_id\nHAVING COUNT(*) > {n}\nORDER BY n DESC\nLIMIT {n};\n",
            f"CREATE TABLE {t} (\n  id INTEGER PRIMARY KEY,\n  {u}_id INTEGER NOT NULL REFERENCES {u}(id),\n"
            f"  label TEXT NOT NULL,\n  weight REAL DEFAULT {n}.0,\n  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n);\n"
            f"CREATE INDEX idx_{t}_{u} ON {t} ({u}_id);\n",
            f"WITH ranked AS (\n  SELECT *, ROW_NUMBER() OVER (PARTITION BY {u}_id ORDER BY created_at DESC) AS rn\n  FROM {t}\n)\n"
            f"SELECT id, {u}_id, label FROM ranked WHERE rn = 1;\n",
            f"UPDATE {t}\nSET weight = weight * 1.{n % 9 + 1}\nWHERE {u}_id IN (SELECT id FROM {u} WHERE active = 1);\n",
        ])

    def _c(self) -> str:
        a = self._ident()
        n = self.rng.randrange(8, 200)
        return self.pick([
            f"#include <stddef.h>\n\nsize_t count_{a}(const int *xs, size_t n, int threshold) {{\n    size_t c = 0;\n"
            f"    for (size_t i = 0; i < n; i++)\n        if (xs[i] > threshold)\n            c++;\n    return c;\n}}\n",
            f"#include <string.h>\n\nvoid reverse_{a}(char *s) {{\n    size_t n = strlen(s);\n    for (size_t i = 0; i < n / 2; i++) {{\n"
            f"        char t = s[i];\n        s[i] = s[n - 1 - i];\n        s[n - 1 - i] = t;\n    }}\n}}\n",
            f"typedef struct {{\n    int items[{n}];\n    int head, tail;\n}} ring_{a};\n\n"
            f"int ring_{a}_push(ring_{a} *r, int v) {{\n    int next = (r->tail + 1) % {n};\n    if (next == r->head)\n        return -1;\n"
            f"    r->items[r->tail] = v;\n    r->tail = next;\n    return 0;\n}}\n",
        ])

    def _sh(self) -> str:
        a = self._ident()
        n = self.rng.randrange(2, 30)
        return self.pick([
            f"#!/usr/bin/env bash\nset -euo pipefail\nfor f in logs/{a}-*.txt; do\n  lines=$(wc -l < \"$f\")\n"
            f"  if [ \"$lines\" -gt {n * 100} ]; then\n    gzip -9 \"$f\"\n  fi\ndone\n",
            f"#!/usr/bin/env bash\nset -eu\ntar -czf backup-$(date +%F).tgz data/{a}\nfind backups -name '*.tgz' -mtime +{n} -delete\n",
        ])

    def code(self, min_chars: int) -> str:
        self.switch_domain()
        parts, size = [], 0
        while size < min_chars:
            r = self.rng.random()
            if r < 0.45:
                c, _ = self.python_module(self.rng.randrange(2, 5))
                c = "# file: " + self._ident() + ".py\n" + c + "\n\n" + self._py_class()
            elif r < 0.65:
                c = "// file: " + self._ident() + ".js\n" + "\n".join(self._js() for _ in range(3))
            elif r < 0.8:
                c = "-- file: " + self._ident() + ".sql\n" + "\n".join(self._sql() for _ in range(3))
            elif r < 0.93:
                c = "/* file: " + self._ident() + ".c */\n" + "\n".join(self._c() for _ in range(2))
            else:
                c = "# file: " + self._ident() + ".sh\n" + self._sh()
            parts.append(c)
            size += len(c) + 2
        return "\n\n".join(parts)

    # ---- records for JSON extraction ------------------------------------------------
    def record(self) -> tuple[str, dict]:
        r = self.rng
        kind = r.randrange(3)
        person = self.name()
        town = self.pick(self.towns)
        if kind == 0:                              # booking
            date = f"2025-{r.randrange(1, 13):02d}-{r.randrange(1, 29):02d}"
            facts = {"guest": person, "city": town, "check_in": date, "nights": r.randrange(1, 12),
                     "rooms": r.randrange(1, 4), "total_eur": r.randrange(80, 2400),
                     "breakfast": bool(r.randrange(2))}
            text = (f"Booking confirmation. Guest: {facts['guest']}. Destination: {town}. Arrival on {date} for "
                    f"{facts['nights']} nights in {facts['rooms']} room(s). Breakfast "
                    f"{'is' if facts['breakfast'] else 'is not'} included. The total comes to {facts['total_eur']} euros.")
        elif kind == 1:                            # invoice
            items = []
            for _ in range(r.randrange(2, 6)):
                items.append({"item": f"{self.pick(self.products)} {self.pick(THINGS)}", "qty": r.randrange(1, 9),
                              "unit_price": r.randrange(3, 400)})
            facts = {"customer": person, "invoice_no": f"INV-{r.randrange(1000, 9999)}", "items": items}
            lines = "; ".join(f"{i['qty']} x {i['item']} at {i['unit_price']} each" for i in items)
            text = f"Invoice {facts['invoice_no']} for {person}. Lines: {lines}."
        else:                                      # profile
            facts = {"name": person, "age": r.randrange(19, 71), "town": town,
                     "languages": [self.pick(["English", "French", "German", "Polish", "Yoruba", "Hindi", "Spanish"])
                                   for _ in range(r.randrange(1, 4))],
                     "role": self.pick(["engineer", "nurse", "teacher", "baker", "pilot", "analyst", "carpenter"])}
            facts["languages"] = sorted(set(facts["languages"]))
            text = (f"{person} is a {facts['age']}-year-old {facts['role']} from {town} who speaks "
                    f"{' and '.join(facts['languages'])}.")
        return text, facts
