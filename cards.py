CARDS = [
    "unknown", "fireball", "zap", "arrows", "tornado", "rocket", "lightning", "freeze",
    "knight", "archers", "goblins", "skeletons", "minions", "bomber", "musketeer",
    "valkyrie", "mini pekka", "giant", "hog rider", "wizard", "witch", "barbarians",
    "pekka", "golem", "lava hound", "sparky", "miner", "princess", "ice wizard",
    "royal ghost", "bandit", "fisherman", "lumberjack", "inferno dragon", "night witch",
    "magic archer", "ram rider", "mega knight", "mother witch", "electro wizard",
    "elite barbarians", "royal giant", "three musketeers", "dark prince", "guards",
    "goblin gang", "minion horde", "rascals", "royal hogs", "zappies", "flying machine",
    "battle ram", "goblin hut", "barbarian hut", "tombstone", "furnace", "inferno tower",
    "bomb tower", "cannon", "tesla", "x-bow", "mortar", "elixir collector",
    "the log", "graveyard", "ice spirit", "fire spirit", "heal spirit", "electro spirit",
    "bats", "wall breakers", "royal delivery", "goblin barrel", "skeleton barrel",
    "clone", "rage", "mirror", "poison", "earthquake", "royal recruits", "archers queen",
    "golden knight", "skeleton king", "mighty miner", "little prince"
]

CARD_TO_ID = {name: i for i, name in enumerate(CARDS)}
ID_TO_CARD = {i: name for i, name in enumerate(CARDS)}
