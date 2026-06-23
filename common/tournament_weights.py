"""
Weight assignments for different tournaments. These weights are used to adjust the importance of matches in the model based on the tournament they belong to.
"""

# Tournament Names

# World Cup and Qualifiers
WC: str = "FIFA World Cup"
WCQ: str = "FIFA World Cup qualification"
WCQ_PLAYOFF: str = "FIFA World Cup qualification (play-off)"
WCQ_CONMEBOL: str = "FIFA World Cup qualification - CONMEBOL"
WCQ_UEFA: str = "FIFA World Cup qualification - UEFA"
WCQ_AFC: str = "FIFA World Cup qualification - AFC"
WCQ_CAF: str = "FIFA World Cup qualification - CAF"
WCQ_CONCACAF: str = "FIFA World Cup qualification - CONCACAF"
WCQ_OFC: str = "FIFA World Cup qualification - OFC"

# Continental Tournaments and Qualifiers
## CONMEBOL
CA: str = "Copa América"
SUPERCLASICO_AMERICAS: str = "Superclásico de las Américas"

## UEFA
EUROS: str = "UEFA Euro"
EUROS_QUAL: str = "UEFA Euro qualification"
UEFA_NL: str = "UEFA Nations League"
BALTIC_CUP: str = "Baltic Cup"
MURATTI_VASE: str = "Muratti Vase"

## CAF
AFCON: str = "African Cup of Nations"
AFCON_QUAL: str = "African Cup of Nations qualification"
MRI_FOUR_NATIONS_CUP: str = "Mauritius Four Nations Cup"

## CONCACAF
GOLD_CUP: str = "Gold Cup"
GOLD_CUP_QUAL: str = "Gold Cup qualification"
CONCACAF_NL: str = "CONCACAF Nations League"
CONCACAF_NL_QUAL: str = "CONCACAF Nations League qualification"

## AFC
AFC_ASIAN_CUP: str = "AFC Asian Cup"
AFC_ASIAN_CUP_QUAL: str = "AFC Asian Cup qualification"

### Arabia & Gulf
ARAB_CUP: str = "Arab Cup"
ARAB_CUP_QUAL: str = "Arab Cup qualification"
GULF_CUP: str = "Gulf Cup"
JOR_INTL_TOURNAMENT: str = "Jordan International Tournament"

### South Asia
SAFF_CUP: str = "SAFF Cup"
TRI_NATION_TOURNAMENT: str = "Tri Nation Tournament"
THREE_NATIONS_CUP: str = "Three Nations Cup"
MAHINDA_RAJAPAKSA_CUP: str = "Mahinda Rajapaksa Cup"

### Southeast Asia
AFF_CHAMPIONSHIP: str = "AFF Championship"
AFF_CHAMPIONSHIP_QUAL: str = "AFF Championship qualification"
KINGS_CUP: str = "King's Cup"

### East Asia
EAFF_CHAMPIONSHIP: str = "EAFF Championship"

### Central Asia
CAFA_NATIONS_CUP: str = "CAFA Nations Cup"
NAVRUZ_CUP: str = "Navruz Cup"

### West Asia
WAFF_CHAMPIONSHIP: str = "WAFF Championship"

## OFC
MSG_PM_CUP: str = "MSG Prime Minister's Cup"
SOC_ASHES: str = "Soccer Ashes"

## Friendly
FRIENDLY: str = "Friendly"

## Interconfederation
INTERCONFEDERATION_CUP: str = "Intercontinental Cup"
FINALISSIMA: str = "CONMEBOL–UEFA Cup of Champions"

## Other
CONIFA_WC: str = "CONIFA World Football Cup"
CONIFA_EURO: str = "CONIFA European Football Cup"
KIRIN_CHALLENGE_CUP: str = "Kirin Challenge Cup"
KIRIN_CUP: str = "Kirin Cup"
COSAFA_CUP: str = "COSAFA Cup"
ISLAND_GAMES: str = "Island Games"
PACIFIC_GAMES: str = "Pacific Games"
INTER_GAMES: str = "Inter Games"
IND_OCEAN_GAMES: str = "Indian Ocean Island Games"


TOURNAMENT_WEIGHTS: dict[str, float] = {
    # Tier 1 - World Cup finals
    WC: 4.0,

    # Tier 2 - Continental championships
    CA: 3.5,
    EUROS: 3.5,
    AFCON: 3.0,
    AFC_ASIAN_CUP: 3.0,
    GOLD_CUP: 2.5,

    # Tier 3 - World Cup qualification
    WCQ: 2.5,
    WCQ_CONMEBOL: 2.5,
    WCQ_UEFA: 2.5,
    WCQ_AFC: 2.5,
    WCQ_CAF: 2.5,
    WCQ_CONCACAF: 2.5,
    WCQ_OFC: 2.5,
    WCQ_PLAYOFF: 3.0,

    # Tier 4 - Continental qualification
    EUROS_QUAL: 2.0,
    AFC_ASIAN_CUP_QUAL: 2.0,
    AFCON_QUAL: 2.0,
    GOLD_CUP_QUAL: 1.8,

    # Nations Leagues
    UEFA_NL: 2.0,
    CONCACAF_NL: 1.8,
    CONCACAF_NL_QUAL: 1.4,

    # Intercontinental
    FINALISSIMA: 2.5,
    INTERCONFEDERATION_CUP: 2.0,

    # Strong regional championships
    ARAB_CUP: 2.0,
    AFF_CHAMPIONSHIP: 1.8,
    EAFF_CHAMPIONSHIP: 1.8,
    CAFA_NATIONS_CUP: 1.8,
    WAFF_CHAMPIONSHIP: 1.7,
    GULF_CUP: 1.7,
    COSAFA_CUP: 1.6,
    SAFF_CUP: 1.5,

    # Regional qualification
    ARAB_CUP_QUAL: 1.4,
    AFF_CHAMPIONSHIP_QUAL: 1.2,

    # Historic bilateral / regional competitions
    SUPERCLASICO_AMERICAS: 1.3,
    SOC_ASHES: 1.0,
    BALTIC_CUP: 1.0,

    # Invitational tournaments
    KIRIN_CUP: 0.8,
    KIRIN_CHALLENGE_CUP: 0.7,
    KINGS_CUP: 0.8,
    JOR_INTL_TOURNAMENT: 0.7,
    MRI_FOUR_NATIONS_CUP: 0.7,
    TRI_NATION_TOURNAMENT: 0.7,
    THREE_NATIONS_CUP: 0.7,
    MAHINDA_RAJAPAKSA_CUP: 0.6,
    NAVRUZ_CUP: 0.7,

    # Small regional multi-sport competitions
    PACIFIC_GAMES: 1.0,
    IND_OCEAN_GAMES: 0.9,
    ISLAND_GAMES: 0.8,
    MSG_PM_CUP: 0.8,
    INTER_GAMES: 0.5,

    # Non-FIFA football
    CONIFA_WC: 0.4,
    CONIFA_EURO: 0.3,

    # Minor local competitions
    MURATTI_VASE: 0.2,

    # Friendlies
    FRIENDLY: 0.5,
}
