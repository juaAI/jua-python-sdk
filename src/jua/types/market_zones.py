"""Market zone definitions and enumerations."""

from dataclasses import dataclass
from enum import Enum

from jua.types.countries import Countries


@dataclass(frozen=True)
class MarketZone:
    """Information about a market zone.

    Attributes:
        name: The market zone identifier/code.
        country: The country enum member.
    """

    name: str
    country: Countries


class MarketZones(Enum):
    """Available market zones for market aggregate queries.

    Each market zone has a name (code) and country enum member.
    Zones are ordered alphabetically by zone code.

    Examples:
        >>> from jua.types import MarketZones, Countries
        >>> zone = MarketZones.DE
        >>> print(zone.zone_name)  # "DE"
        >>> print(zone.country)  # Countries.GERMANY
        >>> print(zone.country.value)  # "Germany"
        >>>
        >>> # Filter zones by country
        >>> german_zones = MarketZones.filter_by_country(Countries.GERMANY)
        >>> print([z.zone_name for z in german_zones])  # ["DE"]
    """

    AD = MarketZone("AD", Countries.ANDORRA)
    AE = MarketZone("AE", Countries.UNITED_ARAB_EMIRATES)
    AF = MarketZone("AF", Countries.AFGHANISTAN)
    AG = MarketZone("AG", Countries.ANTIGUA_AND_BARBUDA)
    AL = MarketZone("AL", Countries.ALBANIA)
    AM = MarketZone("AM", Countries.ARMENIA)
    AO = MarketZone("AO", Countries.ANGOLA)
    AR = MarketZone("AR", Countries.ARGENTINA)
    AT = MarketZone("AT", Countries.AUSTRIA)
    AU_LH = MarketZone("AU-LH", Countries.AUSTRALIA)
    AU_NSW = MarketZone("AU-NSW", Countries.AUSTRALIA)
    AU_NT = MarketZone("AU-NT", Countries.AUSTRALIA)
    AU_QLD = MarketZone("AU-QLD", Countries.AUSTRALIA)
    AU_SA = MarketZone("AU-SA", Countries.AUSTRALIA)
    AU_TAS = MarketZone("AU-TAS", Countries.AUSTRALIA)
    AU_TAS_CBI = MarketZone("AU-TAS-CBI", Countries.AUSTRALIA)
    AU_TAS_FI = MarketZone("AU-TAS-FI", Countries.AUSTRALIA)
    AU_TAS_KI = MarketZone("AU-TAS-KI", Countries.AUSTRALIA)
    AU_VIC = MarketZone("AU-VIC", Countries.AUSTRALIA)
    AU_WA = MarketZone("AU-WA", Countries.AUSTRALIA)
    AW = MarketZone("AW", Countries.ARUBA)
    AX = MarketZone("AX", Countries.ALAND_ISLANDS)
    AZ = MarketZone("AZ", Countries.AZERBAIJAN)
    BA = MarketZone("BA", Countries.BOSNIA_AND_HERZEGOVINA)
    BB = MarketZone("BB", Countries.BARBADOS)
    BD = MarketZone("BD", Countries.BANGLADESH)
    BE = MarketZone("BE", Countries.BELGIUM)
    BF = MarketZone("BF", Countries.BURKINA_FASO)
    BG = MarketZone("BG", Countries.BULGARIA)
    BH = MarketZone("BH", Countries.BAHRAIN)
    BI = MarketZone("BI", Countries.BURUNDI)
    BJ = MarketZone("BJ", Countries.BENIN)
    BM = MarketZone("BM", Countries.BERMUDA)
    BN = MarketZone("BN", Countries.BRUNEI_DARUSSALAM)
    BO = MarketZone("BO", Countries.BOLIVIA_PLURINATIONAL_STATE_OF)
    BR_CS = MarketZone("BR-CS", Countries.BRAZIL)
    BR_N = MarketZone("BR-N", Countries.BRAZIL)
    BR_NE = MarketZone("BR-NE", Countries.BRAZIL)
    BR_S = MarketZone("BR-S", Countries.BRAZIL)
    BS = MarketZone("BS", Countries.BAHAMAS)
    BT = MarketZone("BT", Countries.BHUTAN)
    BW = MarketZone("BW", Countries.BOTSWANA)
    BY = MarketZone("BY", Countries.BELARUS)
    BZ = MarketZone("BZ", Countries.BELIZE)
    CA_AB = MarketZone("CA-AB", Countries.CANADA)
    CA_BC = MarketZone("CA-BC", Countries.CANADA)
    CA_MB = MarketZone("CA-MB", Countries.CANADA)
    CA_NB = MarketZone("CA-NB", Countries.CANADA)
    CA_NL = MarketZone("CA-NL", Countries.CANADA)
    CA_NS = MarketZone("CA-NS", Countries.CANADA)
    CA_NT = MarketZone("CA-NT", Countries.CANADA)
    CA_NU = MarketZone("CA-NU", Countries.CANADA)
    CA_ON = MarketZone("CA-ON", Countries.CANADA)
    CA_PE = MarketZone("CA-PE", Countries.CANADA)
    CA_QC = MarketZone("CA-QC", Countries.CANADA)
    CA_SK = MarketZone("CA-SK", Countries.CANADA)
    CA_YT = MarketZone("CA-YT", Countries.CANADA)
    CD = MarketZone("CD", Countries.CONGO_THE_DEMOCRATIC_REPUBLIC_OF_THE)
    CF = MarketZone("CF", Countries.CENTRAL_AFRICAN_REPUBLIC)
    CG = MarketZone("CG", Countries.CONGO)
    CH = MarketZone("CH", Countries.SWITZERLAND)
    CI = MarketZone("CI", Countries.COTE_DIVOIRE)
    CL_CHP = MarketZone("CL-CHP", Countries.CHILE)
    CL_SEA = MarketZone("CL-SEA", Countries.CHILE)
    CL_SEM = MarketZone("CL-SEM", Countries.CHILE)
    CL_SEN = MarketZone("CL-SEN", Countries.CHILE)
    CM = MarketZone("CM", Countries.CAMEROON)
    CN = MarketZone("CN", Countries.CHINA)
    CO = MarketZone("CO", Countries.COLOMBIA)
    CR = MarketZone("CR", Countries.COSTA_RICA)
    CU = MarketZone("CU", Countries.CUBA)
    CV = MarketZone("CV", Countries.CABO_VERDE)
    CW = MarketZone("CW", Countries.CURACAO)
    CY = MarketZone("CY", Countries.CYPRUS)
    CZ = MarketZone("CZ", Countries.CZECHIA)
    DE = MarketZone("DE", Countries.GERMANY)
    DJ = MarketZone("DJ", Countries.DJIBOUTI)
    DK_BHM = MarketZone("DK-BHM", Countries.DENMARK)
    DK_DK1 = MarketZone("DK-DK1", Countries.DENMARK)
    DK_DK2 = MarketZone("DK-DK2", Countries.DENMARK)
    DM = MarketZone("DM", Countries.DOMINICA)
    DO = MarketZone("DO", Countries.DOMINICAN_REPUBLIC)
    DZ = MarketZone("DZ", Countries.ALGERIA)
    EC = MarketZone("EC", Countries.ECUADOR)
    EE = MarketZone("EE", Countries.ESTONIA)
    EG = MarketZone("EG", Countries.EGYPT)
    EH = MarketZone("EH", Countries.WESTERN_SAHARA)
    ER = MarketZone("ER", Countries.ERITREA)
    ES = MarketZone("ES", Countries.SPAIN)
    ES_CN_FV = MarketZone("ES-CN-FV", Countries.SPAIN)
    ES_CN_GC = MarketZone("ES-CN-GC", Countries.SPAIN)
    ES_CN_HI = MarketZone("ES-CN-HI", Countries.SPAIN)
    ES_CN_IG = MarketZone("ES-CN-IG", Countries.SPAIN)
    ES_CN_LP = MarketZone("ES-CN-LP", Countries.SPAIN)
    ES_CN_LZ = MarketZone("ES-CN-LZ", Countries.SPAIN)
    ES_CN_TE = MarketZone("ES-CN-TE", Countries.SPAIN)
    ES_IB_FO = MarketZone("ES-IB-FO", Countries.SPAIN)
    ES_IB_IZ = MarketZone("ES-IB-IZ", Countries.SPAIN)
    ES_IB_MA = MarketZone("ES-IB-MA", Countries.SPAIN)
    ES_IB_ME = MarketZone("ES-IB-ME", Countries.SPAIN)
    ET = MarketZone("ET", Countries.ETHIOPIA)
    FI = MarketZone("FI", Countries.FINLAND)
    FJ = MarketZone("FJ", Countries.FIJI)
    FK = MarketZone("FK", Countries.FALKLAND_ISLANDS_MALVINAS)
    FM = MarketZone("FM", Countries.MICRONESIA_FEDERATED_STATES_OF)
    FO_MI = MarketZone("FO-MI", Countries.FAROE_ISLANDS)
    FO_SI = MarketZone("FO-SI", Countries.FAROE_ISLANDS)
    FR = MarketZone("FR", Countries.FRANCE)
    FR_COR = MarketZone("FR-COR", Countries.FRANCE)
    GA = MarketZone("GA", Countries.GABON)
    GB = MarketZone("GB", Countries.UNITED_KINGDOM)
    GB_NIR = MarketZone("GB-NIR", Countries.UNITED_KINGDOM)
    GB_ORK = MarketZone("GB-ORK", Countries.UNITED_KINGDOM)
    GB_ZET = MarketZone("GB-ZET", Countries.UNITED_KINGDOM)
    GE = MarketZone("GE", Countries.GEORGIA)
    GF = MarketZone("GF", Countries.FRENCH_GUIANA)
    GG = MarketZone("GG", Countries.GUERNSEY)
    GH = MarketZone("GH", Countries.GHANA)
    GL = MarketZone("GL", Countries.GREENLAND)
    GM = MarketZone("GM", Countries.GAMBIA)
    GN = MarketZone("GN", Countries.GUINEA)
    GP = MarketZone("GP", Countries.GUADELOUPE)
    GQ = MarketZone("GQ", Countries.EQUATORIAL_GUINEA)
    GR = MarketZone("GR", Countries.GREECE)
    GT = MarketZone("GT", Countries.GUATEMALA)
    GU = MarketZone("GU", Countries.GUAM)
    GW = MarketZone("GW", Countries.GUINEA_BISSAU)
    GY = MarketZone("GY", Countries.GUYANA)
    HK = MarketZone("HK", Countries.HONG_KONG)
    HN = MarketZone("HN", Countries.HONDURAS)
    HR = MarketZone("HR", Countries.CROATIA)
    HT = MarketZone("HT", Countries.HAITI)
    HU = MarketZone("HU", Countries.HUNGARY)
    ID = MarketZone("ID", Countries.INDONESIA)
    IE = MarketZone("IE", Countries.IRELAND)
    IL = MarketZone("IL", Countries.ISRAEL)
    IM = MarketZone("IM", Countries.ISLE_OF_MAN)
    IN_AN = MarketZone("IN-AN", Countries.INDIA)
    IN_EA = MarketZone("IN-EA", Countries.INDIA)
    IN_NE = MarketZone("IN-NE", Countries.INDIA)
    IN_NO = MarketZone("IN-NO", Countries.INDIA)
    IN_SO = MarketZone("IN-SO", Countries.INDIA)
    IN_WE = MarketZone("IN-WE", Countries.INDIA)
    IQ = MarketZone("IQ", Countries.IRAQ)
    IR = MarketZone("IR", Countries.IRAN_ISLAMIC_REPUBLIC_OF)
    IS = MarketZone("IS", Countries.ICELAND)
    IT_CNO = MarketZone("IT-CNO", Countries.ITALY)
    IT_CSO = MarketZone("IT-CSO", Countries.ITALY)
    IT_NO = MarketZone("IT-NO", Countries.ITALY)
    IT_SAR = MarketZone("IT-SAR", Countries.ITALY)
    IT_SIC = MarketZone("IT-SIC", Countries.ITALY)
    IT_SO = MarketZone("IT-SO", Countries.ITALY)
    JE = MarketZone("JE", Countries.JERSEY)
    JM = MarketZone("JM", Countries.JAMAICA)
    JO = MarketZone("JO", Countries.JORDAN)
    JP_CB = MarketZone("JP-CB", Countries.JAPAN)
    JP_CG = MarketZone("JP-CG", Countries.JAPAN)
    JP_HKD = MarketZone("JP-HKD", Countries.JAPAN)
    JP_HR = MarketZone("JP-HR", Countries.JAPAN)
    JP_KN = MarketZone("JP-KN", Countries.JAPAN)
    JP_KY = MarketZone("JP-KY", Countries.JAPAN)
    JP_ON = MarketZone("JP-ON", Countries.JAPAN)
    JP_SK = MarketZone("JP-SK", Countries.JAPAN)
    JP_TH = MarketZone("JP-TH", Countries.JAPAN)
    JP_TK = MarketZone("JP-TK", Countries.JAPAN)
    KE = MarketZone("KE", Countries.KENYA)
    KG = MarketZone("KG", Countries.KYRGYZSTAN)
    KH = MarketZone("KH", Countries.CAMBODIA)
    KM = MarketZone("KM", Countries.COMOROS)
    KP = MarketZone("KP", Countries.KOREA_DEMOCRATIC_PEOPLES_REPUBLIC_OF)
    KR = MarketZone("KR", Countries.KOREA_REPUBLIC_OF)
    KW = MarketZone("KW", Countries.KUWAIT)
    KY = MarketZone("KY", Countries.CAYMAN_ISLANDS)
    KZ = MarketZone("KZ", Countries.KAZAKHSTAN)
    LA = MarketZone("LA", Countries.LAO_PEOPLES_DEMOCRATIC_REPUBLIC)
    LB = MarketZone("LB", Countries.LEBANON)
    LC = MarketZone("LC", Countries.SAINT_LUCIA)
    LI = MarketZone("LI", Countries.LIECHTENSTEIN)
    LK = MarketZone("LK", Countries.SRI_LANKA)
    LR = MarketZone("LR", Countries.LIBERIA)
    LS = MarketZone("LS", Countries.LESOTHO)
    LT = MarketZone("LT", Countries.LITHUANIA)
    LU = MarketZone("LU", Countries.LUXEMBOURG)
    LV = MarketZone("LV", Countries.LATVIA)
    LY = MarketZone("LY", Countries.LIBYA)
    MA = MarketZone("MA", Countries.MOROCCO)
    MD = MarketZone("MD", Countries.MOLDOVA_REPUBLIC_OF)
    ME = MarketZone("ME", Countries.MONTENEGRO)
    MG = MarketZone("MG", Countries.MADAGASCAR)
    MK = MarketZone("MK", Countries.NORTH_MACEDONIA)
    ML = MarketZone("ML", Countries.MALI)
    MM = MarketZone("MM", Countries.MYANMAR)
    MN = MarketZone("MN", Countries.MONGOLIA)
    MQ = MarketZone("MQ", Countries.MARTINIQUE)
    MR = MarketZone("MR", Countries.MAURITANIA)
    MT = MarketZone("MT", Countries.MALTA)
    MU = MarketZone("MU", Countries.MAURITIUS)
    MV = MarketZone("MV", Countries.MALDIVES)
    MW = MarketZone("MW", Countries.MALAWI)
    MX = MarketZone("MX", Countries.MEXICO)
    MY_EM = MarketZone("MY-EM", Countries.MALAYSIA)
    MY_WM = MarketZone("MY-WM", Countries.MALAYSIA)
    MZ = MarketZone("MZ", Countries.MOZAMBIQUE)
    NA = MarketZone("NA", Countries.NAMIBIA)
    NC = MarketZone("NC", Countries.NEW_CALEDONIA)
    NE = MarketZone("NE", Countries.NIGER)
    NG = MarketZone("NG", Countries.NIGERIA)
    NI = MarketZone("NI", Countries.NICARAGUA)
    NL = MarketZone("NL", Countries.NETHERLANDS)
    NO_NO1 = MarketZone("NO-NO1", Countries.NORWAY)
    NO_NO2 = MarketZone("NO-NO2", Countries.NORWAY)
    NO_NO3 = MarketZone("NO-NO3", Countries.NORWAY)
    NO_NO4 = MarketZone("NO-NO4", Countries.NORWAY)
    NO_NO5 = MarketZone("NO-NO5", Countries.NORWAY)
    NP = MarketZone("NP", Countries.NEPAL)
    NZ = MarketZone("NZ", Countries.NEW_ZEALAND)
    NZ_NZC = MarketZone("NZ-NZC", Countries.NEW_ZEALAND)
    NZ_NZST = MarketZone("NZ-NZST", Countries.NEW_ZEALAND)
    OM = MarketZone("OM", Countries.OMAN)
    PA = MarketZone("PA", Countries.PANAMA)
    PE = MarketZone("PE", Countries.PERU)
    PF = MarketZone("PF", Countries.FRENCH_POLYNESIA)
    PG = MarketZone("PG", Countries.PAPUA_NEW_GUINEA)
    PH_LU = MarketZone("PH-LU", Countries.PHILIPPINES)
    PH_MI = MarketZone("PH-MI", Countries.PHILIPPINES)
    PH_VI = MarketZone("PH-VI", Countries.PHILIPPINES)
    PK = MarketZone("PK", Countries.PAKISTAN)
    PL = MarketZone("PL", Countries.POLAND)
    PM = MarketZone("PM", Countries.SAINT_PIERRE_AND_MIQUELON)
    PR = MarketZone("PR", Countries.PUERTO_RICO)
    PS = MarketZone("PS", Countries.PALESTINE_STATE_OF)
    PT = MarketZone("PT", Countries.PORTUGAL)
    PT_AC = MarketZone("PT-AC", Countries.PORTUGAL)
    PT_MA = MarketZone("PT-MA", Countries.PORTUGAL)
    PW = MarketZone("PW", Countries.PALAU)
    PY = MarketZone("PY", Countries.PARAGUAY)
    QA = MarketZone("QA", Countries.QATAR)
    RE = MarketZone("RE", Countries.REUNION)
    RO = MarketZone("RO", Countries.ROMANIA)
    RS = MarketZone("RS", Countries.SERBIA)
    RU_1 = MarketZone("RU-1", Countries.RUSSIAN_FEDERATION)
    RU_2 = MarketZone("RU-2", Countries.RUSSIAN_FEDERATION)
    RU_AS = MarketZone("RU-AS", Countries.RUSSIAN_FEDERATION)
    RU_EU = MarketZone("RU-EU", Countries.RUSSIAN_FEDERATION)
    RU_FE = MarketZone("RU-FE", Countries.RUSSIAN_FEDERATION)
    RU_KGD = MarketZone("RU-KGD", Countries.RUSSIAN_FEDERATION)
    RW = MarketZone("RW", Countries.RWANDA)
    SA = MarketZone("SA", Countries.SAUDI_ARABIA)
    SB = MarketZone("SB", Countries.SOLOMON_ISLANDS)
    SC = MarketZone("SC", Countries.SEYCHELLES)
    SD = MarketZone("SD", Countries.SUDAN)
    SE_SE1 = MarketZone("SE-SE1", Countries.SWEDEN)
    SE_SE2 = MarketZone("SE-SE2", Countries.SWEDEN)
    SE_SE3 = MarketZone("SE-SE3", Countries.SWEDEN)
    SE_SE4 = MarketZone("SE-SE4", Countries.SWEDEN)
    SG = MarketZone("SG", Countries.SINGAPORE)
    SI = MarketZone("SI", Countries.SLOVENIA)
    SJ = MarketZone("SJ", Countries.SVALBARD_AND_JAN_MAYEN)
    SK = MarketZone("SK", Countries.SLOVAKIA)
    SL = MarketZone("SL", Countries.SIERRA_LEONE)
    SN = MarketZone("SN", Countries.SENEGAL)
    SO = MarketZone("SO", Countries.SOMALIA)
    SR = MarketZone("SR", Countries.SURINAME)
    SS = MarketZone("SS", Countries.SOUTH_SUDAN)
    ST = MarketZone("ST", Countries.SAO_TOME_AND_PRINCIPE)
    SV = MarketZone("SV", Countries.EL_SALVADOR)
    SY = MarketZone("SY", Countries.SYRIAN_ARAB_REPUBLIC)
    SZ = MarketZone("SZ", Countries.ESWATINI)
    TD = MarketZone("TD", Countries.CHAD)
    TG = MarketZone("TG", Countries.TOGO)
    TH = MarketZone("TH", Countries.THAILAND)
    TJ = MarketZone("TJ", Countries.TAJIKISTAN)
    TL = MarketZone("TL", Countries.TIMOR_LESTE)
    TM = MarketZone("TM", Countries.TURKMENISTAN)
    TN = MarketZone("TN", Countries.TUNISIA)
    TO = MarketZone("TO", Countries.TONGA)
    TR = MarketZone("TR", Countries.TURKEY)
    TT = MarketZone("TT", Countries.TRINIDAD_AND_TOBAGO)
    TW = MarketZone("TW", Countries.TAIWAN_PROVINCE_OF_CHINA)
    TZ = MarketZone("TZ", Countries.TANZANIA_UNITED_REPUBLIC_OF)
    UA = MarketZone("UA", Countries.UKRAINE)
    UA_CR = MarketZone("UA-CR", Countries.UKRAINE)
    UG = MarketZone("UG", Countries.UGANDA)
    US_AK = MarketZone("US-AK", Countries.UNITED_STATES)
    US_AK_SEAPA = MarketZone("US-AK-SEAPA", Countries.UNITED_STATES)
    US_CAL_BANC = MarketZone("US-CAL-BANC", Countries.UNITED_STATES)
    US_CAL_CISO = MarketZone("US-CAL-CISO", Countries.UNITED_STATES)
    US_CAL_IID = MarketZone("US-CAL-IID", Countries.UNITED_STATES)
    US_CAL_LDWP = MarketZone("US-CAL-LDWP", Countries.UNITED_STATES)
    US_CAL_TIDC = MarketZone("US-CAL-TIDC", Countries.UNITED_STATES)
    US_CAR_CPLE = MarketZone("US-CAR-CPLE", Countries.UNITED_STATES)
    US_CAR_CPLW = MarketZone("US-CAR-CPLW", Countries.UNITED_STATES)
    US_CAR_DUK = MarketZone("US-CAR-DUK", Countries.UNITED_STATES)
    US_CAR_SC = MarketZone("US-CAR-SC", Countries.UNITED_STATES)
    US_CAR_SCEG = MarketZone("US-CAR-SCEG", Countries.UNITED_STATES)
    US_CENT_SPA = MarketZone("US-CENT-SPA", Countries.UNITED_STATES)
    US_CENT_SWPP = MarketZone("US-CENT-SWPP", Countries.UNITED_STATES)
    US_FLA_FMPP = MarketZone("US-FLA-FMPP", Countries.UNITED_STATES)
    US_FLA_FPC = MarketZone("US-FLA-FPC", Countries.UNITED_STATES)
    US_FLA_FPL = MarketZone("US-FLA-FPL", Countries.UNITED_STATES)
    US_FLA_GVL = MarketZone("US-FLA-GVL", Countries.UNITED_STATES)
    US_FLA_HST = MarketZone("US-FLA-HST", Countries.UNITED_STATES)
    US_FLA_JEA = MarketZone("US-FLA-JEA", Countries.UNITED_STATES)
    US_FLA_SEC = MarketZone("US-FLA-SEC", Countries.UNITED_STATES)
    US_FLA_TAL = MarketZone("US-FLA-TAL", Countries.UNITED_STATES)
    US_FLA_TEC = MarketZone("US-FLA-TEC", Countries.UNITED_STATES)
    US_HI = MarketZone("US-HI", Countries.UNITED_STATES)
    US_MIDA_PJM = MarketZone("US-MIDA-PJM", Countries.UNITED_STATES)
    US_MIDW_AECI = MarketZone("US-MIDW-AECI", Countries.UNITED_STATES)
    US_MIDW_LGEE = MarketZone("US-MIDW-LGEE", Countries.UNITED_STATES)
    US_MIDW_MISO = MarketZone("US-MIDW-MISO", Countries.UNITED_STATES)
    US_NE_ISNE = MarketZone("US-NE-ISNE", Countries.UNITED_STATES)
    US_NW_AVA = MarketZone("US-NW-AVA", Countries.UNITED_STATES)
    US_NW_BPAT = MarketZone("US-NW-BPAT", Countries.UNITED_STATES)
    US_NW_CHPD = MarketZone("US-NW-CHPD", Countries.UNITED_STATES)
    US_NW_DOPD = MarketZone("US-NW-DOPD", Countries.UNITED_STATES)
    US_NW_GCPD = MarketZone("US-NW-GCPD", Countries.UNITED_STATES)
    US_NW_IPCO = MarketZone("US-NW-IPCO", Countries.UNITED_STATES)
    US_NW_NEVP = MarketZone("US-NW-NEVP", Countries.UNITED_STATES)
    US_NW_NWMT = MarketZone("US-NW-NWMT", Countries.UNITED_STATES)
    US_NW_PACE = MarketZone("US-NW-PACE", Countries.UNITED_STATES)
    US_NW_PACW = MarketZone("US-NW-PACW", Countries.UNITED_STATES)
    US_NW_PGE = MarketZone("US-NW-PGE", Countries.UNITED_STATES)
    US_NW_PSCO = MarketZone("US-NW-PSCO", Countries.UNITED_STATES)
    US_NW_PSEI = MarketZone("US-NW-PSEI", Countries.UNITED_STATES)
    US_NW_SCL = MarketZone("US-NW-SCL", Countries.UNITED_STATES)
    US_NW_TPWR = MarketZone("US-NW-TPWR", Countries.UNITED_STATES)
    US_NW_WACM = MarketZone("US-NW-WACM", Countries.UNITED_STATES)
    US_NW_WAUW = MarketZone("US-NW-WAUW", Countries.UNITED_STATES)
    US_NY_NYIS = MarketZone("US-NY-NYIS", Countries.UNITED_STATES)
    US_SE_SOCO = MarketZone("US-SE-SOCO", Countries.UNITED_STATES)
    US_SW_AZPS = MarketZone("US-SW-AZPS", Countries.UNITED_STATES)
    US_SW_EPE = MarketZone("US-SW-EPE", Countries.UNITED_STATES)
    US_SW_PNM = MarketZone("US-SW-PNM", Countries.UNITED_STATES)
    US_SW_SRP = MarketZone("US-SW-SRP", Countries.UNITED_STATES)
    US_SW_TEPC = MarketZone("US-SW-TEPC", Countries.UNITED_STATES)
    US_SW_WALC = MarketZone("US-SW-WALC", Countries.UNITED_STATES)
    US_TEN_TVA = MarketZone("US-TEN-TVA", Countries.UNITED_STATES)
    US_TEX_ERCO = MarketZone("US-TEX-ERCO", Countries.UNITED_STATES)
    UY = MarketZone("UY", Countries.URUGUAY)
    UZ = MarketZone("UZ", Countries.UZBEKISTAN)
    VC = MarketZone("VC", Countries.SAINT_VINCENT_AND_THE_GRENADINES)
    VE = MarketZone("VE", Countries.VENEZUELA_BOLIVARIAN_REPUBLIC_OF)
    VI = MarketZone("VI", Countries.VIRGIN_ISLANDS_U_S)
    VN = MarketZone("VN", Countries.VIET_NAM)
    VU = MarketZone("VU", Countries.VANUATU)
    WS = MarketZone("WS", Countries.SAMOA)
    XK = MarketZone("XK", Countries.KOSOVO)
    XX = MarketZone("XX", Countries.CYPRUS)
    YE = MarketZone("YE", Countries.YEMEN)
    YT = MarketZone("YT", Countries.MAYOTTE)
    ZA = MarketZone("ZA", Countries.SOUTH_AFRICA)
    ZM = MarketZone("ZM", Countries.ZAMBIA)
    ZW = MarketZone("ZW", Countries.ZIMBABWE)

    @property
    def zone_name(self) -> str:
        """Get the market zone code/name.

        Returns:
            The market zone identifier.
        """
        return self.value.name

    @property
    def country(self) -> Countries:
        """Get the country enum member.

        Returns:
            The Countries enum member.
        """
        return self.value.country

    @property
    def country_name(self) -> str:
        """Get the country name as a string.

        Returns:
            The country name.
        """
        return self.value.country.value

    def __str__(self) -> str:
        """Get string representation (returns the zone name)."""
        return self.value.name

    @classmethod
    def filter_by_country(cls, country: Countries | str) -> list["MarketZones"]:
        """Returns the market zones for a country.

        Args:
            country: The country to get the market zones for.

        Returns:
            List of MarketZones for the specified country.

        Examples:
            >>> from jua.types import MarketZones, Countries
            >>> zones = MarketZones.get(Countries.NORWAY)
            >>> print([z.zone_name for z in zones])
            ['NO-NO1', 'NO-NO2', 'NO-NO3', 'NO-NO4', 'NO-NO5']
            >>>
            >>> # Can also use string
            >>> zones = MarketZones.get("Norway")
        """
        # If it's a Countries enum, use it directly; otherwise convert string to enum
        if isinstance(country, Countries):
            return [zone for zone in cls if zone.country == country]
        else:
            # Find by country name string
            country_str = str(country)
            return [zone for zone in cls if zone.country.value == country_str]
