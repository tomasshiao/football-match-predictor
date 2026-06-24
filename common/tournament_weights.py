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
UNITY_CUP: str = "Unity Cup"

## CAF
AFCON: str = "African Cup of Nations"
AFCON_QUAL: str = "African Cup of Nations qualification"
MRI_FOUR_NATIONS_CUP: str = "Mauritius Four Nations Cup"
MAPINDUZI_CUP: str = "Mapinduzi Cup"
MAR_AFRICAN_FOOTBALL: str = "Morocco, Capital of African Football"
MUKURU: str = "Mukuru 4 Nations"

## CONCACAF
GOLD_CUP: str = "Gold Cup"
GOLD_CUP_QUAL: str = "Gold Cup qualification"
CONCACAF_NL: str = "CONCACAF Nations League"
CONCACAF_NL_QUAL: str = "CONCACAF Nations League qualification"
CAN_SHIELD: str = "Canadian Shield"
CONCACAF_SERIES: str = "CONCACAF Series"
CA_QUALI: str = "Copa América qualification"

## AFC
AFC_ASIAN_CUP: str = "AFC Asian Cup"
AFC_ASIAN_CUP_QUAL: str = "AFC Asian Cup qualification"

### Arabia & Gulf
ARAB_CUP: str = "Arab Cup"
ARAB_CUP_QUAL: str = "Arab Cup qualification"
GULF_CUP: str = "Gulf Cup"
JOR_INTL_TOURNAMENT: str = "Jordan International Tournament"
AL_AIN_INTL_CUP: str = "Al Ain International Cup"

### South Asia
SAFF_CUP: str = "SAFF Cup"
TRI_NATION_TOURNAMENT: str = "Tri Nation Tournament"
THREE_NATIONS_CUP: str = "Three Nations Cup"
MAHINDA_RAJAPAKSA_CUP: str = "Mahinda Rajapaksa Cup"
DIAMOND_JUBILEE_CUP: str = "Diamond Jubilee International Football Tournament"
TRI_NATION_CUP: str = "Tri-Nations Cup"
TRI_NATIONS_SERIES: str = "Tri-Nations Series"
SA_SUPER_CUP: str = "South Asian Super Cup"

### Southeast Asia
AFF_CHAMPIONSHIP: str = "AFF Championship"
AFF_CHAMPIONSHIP_QUAL: str = "AFF Championship qualification"
KINGS_CUP: str = "King's Cup"
MERDEKA_CUP: str = "Merdeka Cup"
MERDEKA_TOURNAMENT: str = "Merdeka Tournament"
ASEAN_CHAMPIONSHIP: str = "ASEAN Championship"
ASEAN_CHAMPIONSHIP_QUAL: str = "ASEAN Championship qualification"

### East Asia
EAFF_CHAMPIONSHIP: str = "EAFF Championship"
EAFF_CHAMPIONSHIP_QUAL: str = "EAFF Championship qualification"

### Central Asia
CAFA_NATIONS_CUP: str = "CAFA Nations Cup"
NAVRUZ_CUP: str = "Navruz Cup"

### West Asia
WAFF_CHAMPIONSHIP: str = "WAFF Championship"

## OFC
OFC_NC: str = "Oceania Nations Cup"
OUTRIGGER_CHALLENGE_CUP: str = "Outrigger Challenge Cup" # MHL
MSG_PM_CUP: str = "MSG Prime Minister's Cup"
SOC_ASHES: str = "Soccer Ashes"

## Friendly
FRIENDLY: str = "Friendly"

## Interconfederation
FIFA_SERIES: str = "FIFA Series"
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
    OFC_NC: 2.2,

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
    CA_QUALI: 1.8,
    ASEAN_CHAMPIONSHIP_QUAL: 1.2,
    EAFF_CHAMPIONSHIP_QUAL: 1.2,

    # Nations Leagues
    UEFA_NL: 2.0,
    CONCACAF_NL: 1.8,
    CONCACAF_NL_QUAL: 1.4,

    # Intercontinental competitions
    FINALISSIMA: 2.5,
    INTERCONFEDERATION_CUP: 2.0,
    FIFA_SERIES: 0.9,

    # Strong regional championships
    ARAB_CUP: 2.0,
    AFF_CHAMPIONSHIP: 1.8,
    ASEAN_CHAMPIONSHIP: 1.8,
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
    CAN_SHIELD: 1.0,

    # Invitational / friendly tournaments
    KIRIN_CUP: 0.8,
    KIRIN_CHALLENGE_CUP: 0.7,
    KINGS_CUP: 0.8,
    MERDEKA_CUP: 0.9,
    MERDEKA_TOURNAMENT: 0.9,
    JOR_INTL_TOURNAMENT: 0.7,
    AL_AIN_INTL_CUP: 0.7,
    MRI_FOUR_NATIONS_CUP: 0.7,
    TRI_NATION_TOURNAMENT: 0.7,
    TRI_NATION_CUP: 0.7,
    TRI_NATIONS_SERIES: 0.7,
    THREE_NATIONS_CUP: 0.7,
    MAHINDA_RAJAPAKSA_CUP: 0.6,
    DIAMOND_JUBILEE_CUP: 0.6,
    NAVRUZ_CUP: 0.7,
    MUKURU: 0.7,
    CONCACAF_SERIES: 0.8,

    # Small regional multi-sport competitions
    PACIFIC_GAMES: 1.0,
    IND_OCEAN_GAMES: 0.9,
    ISLAND_GAMES: 0.8,
    MSG_PM_CUP: 0.8,
    OUTRIGGER_CHALLENGE_CUP: 0.8,
    INTER_GAMES: 0.5,

    # Special / promotional tournaments
    MAPINDUZI_CUP: 0.5,
    MAR_AFRICAN_FOOTBALL: 0.5,
    UNITY_CUP: 0.8,
    SA_SUPER_CUP: 0.8,

    # Non-FIFA football
    CONIFA_WC: 0.4,
    CONIFA_EURO: 0.3,

    # Minor local competitions
    MURATTI_VASE: 0.2,

    # Friendlies
    FRIENDLY: 0.5,
}