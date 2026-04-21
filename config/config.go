package config

type CityConfig struct {
	Bridges         int
	MaxLines        int
	StationCapacity int
	TrainCapacity   int
}

type PointF struct {
	X float64
	Y float64
}

const (
	StationRadius                = 15.0
	PassengerSize                = 6.0
	TrainWidth                   = 25.0
	TrainHeight                  = 15.0
	LineWidth                    = 6.0
	MaxPassengersPerStation      = 6
	MaxPassengersWithInterchange = 18
	TrainCapacity                = 6
	MaxTrainsPerLine             = 4
	MaxCarriagesPerTrain         = 3
	TrainMaxSpeed                = 1.2
	TrainAcceleration            = 0.002
	PassengerBoardTime           = 200
	InterchangeTransferTime      = 50
	RegularTransferTime          = 400
	PassengerWaitPatience        = 10000
	BaseSpawnRate                = 2000
	BaseStationSpawnRate         = 15000
	DifficultyScaleFactor        = 0.8
	OvercrowdTime                = 45000
	WeekDuration                 = 60000

	// Overcrowding thresholds and grace periods
	OvercrowdCriticalThreshold = 0.85  // fraction of OvercrowdTime that triggers AI emergency
	OvercrowdRepathThreshold   = 0.8   // fraction of OvercrowdTime that triggers eager repath
	OvercrowdGraceExtra        = 2000  // extra ms when a train is approaching the overcrowded station
	RepathCooldownMs           = 2000  // minimum interval between forced repath attempts per passenger

	// Pathfinding scoring weights
	TransferPenalty        = 2.5  // extra score per line change in BFS
	OvercrowdScoreFactor   = 4.0  // score penalty multiplier for overcrowded intermediate stations

	// Solver timing
	GhostLineTimeoutMs = 8000 // ms before an unused ghost line is discarded

	// Solver interval timing (milliseconds)
	SolverEmergencyIntervalMs   = 80.0  // run interval when any station exceeds OvercrowdCriticalThreshold
	SolverUrgentIntervalMs      = 150.0 // run interval when any station exceeds SolverUrgentOvercrowdFrac
	SolverUrgentOvercrowdFrac   = 0.6   // overcrowd fraction that triggers the urgent (150 ms) interval

	// Solver upgrade scoring weights
	SolverIsolatedStationPts    = 25.0 // score per isolated station when evaluating NewLine
	SolverNewLineOvercrowdWt    = 0.04 // overcrowd progress weight for NewLine
	SolverNewLineArcWt          = 0.06 // avg arc length weight for NewLine
	SolverNewLineBase           = 18.0 // baseline score so NewLine is competitive early on
	SolverCarriageCapWt         = 20.0 // capacity ratio multiplier for Carriage upgrade
	SolverCarriageOvercrowdWt   = 0.03 // overcrowd progress weight for Carriage upgrade
	SolverMinCarriageFillRatio  = 0.6  // minimum fill ratio to trigger carriage assignment
	SolverInterchangeLinesWt    = 8.0  // score per extra line at interchange candidate
	SolverInterchangeOvercrowdWt = 0.1 // overcrowd weight for interchange score
	SolverUrgencyOvercrowdWt    = 0.02 // per-station urgency weight in placement scoring

	// Station placement distances (pixels)
	StationMinDistance = 120.0 // minimum distance between stations at init and spawn

	// Rendering
	OvercrowdPulseHz = 150.0 // divisor used in sin(nowMs/OvercrowdPulseHz) for station pulse

	// Station passenger queue safety cap. Prevents unbounded memory growth when
	// a station is completely disconnected for many weeks. Set well above any
	// realistic interchange capacity (MaxPassengersWithInterchange = 18) so it
	// only activates in degenerate cases.
	MaxPassengerQueueCap = 100

	// Fraction of min(screenWidth, screenHeight) used as the radius when placing
	// initial stations. Keeps stations clustered near the map centre at game start.
	InitialStationSpreadFraction = 0.2
)

var LineColors = []string{
	"#E45E4F", "#4A8CBA", "#F1A42B", "#5AA96B", "#9B6FBA", "#3ABDB0", "#E8844A",
}

var Cities = map[string]CityConfig{
	"london":    {Bridges: 2, MaxLines: 5, StationCapacity: 6, TrainCapacity: 6},
	"paris":     {Bridges: 3, MaxLines: 6, StationCapacity: 4, TrainCapacity: 6},
	"newyork":   {Bridges: 2, MaxLines: 7, StationCapacity: 6, TrainCapacity: 6},
	"tokyo":     {Bridges: 4, MaxLines: 6, StationCapacity: 6, TrainCapacity: 6},
	"cairo":     {Bridges: 2, MaxLines: 5, StationCapacity: 6, TrainCapacity: 4},
	"mumbai":    {Bridges: 2, MaxLines: 5, StationCapacity: 6, TrainCapacity: 4},
	"hongkong":  {Bridges: 3, MaxLines: 6, StationCapacity: 6, TrainCapacity: 6},
	"osaka":     {Bridges: 2, MaxLines: 6, StationCapacity: 6, TrainCapacity: 6},
	"melbourne": {Bridges: 2, MaxLines: 5, StationCapacity: 6, TrainCapacity: 6},
	"saopaulo":  {Bridges: 1, MaxLines: 5, StationCapacity: 6, TrainCapacity: 6},
}

var Rivers = map[string][][]PointF{
	"london": {
		{
			{X: 0.0, Y: 0.65}, {X: 0.3, Y: 0.55},
			{X: 0.7, Y: 0.60}, {X: 1.0, Y: 0.50},
			{X: 1.0, Y: 0.60}, {X: 0.7, Y: 0.70},
			{X: 0.3, Y: 0.65}, {X: 0.0, Y: 0.75},
		},
	},
	"paris": {
		{
			{X: 0.0, Y: 0.44}, {X: 0.25, Y: 0.41},
			{X: 0.50, Y: 0.49}, {X: 0.75, Y: 0.46},
			{X: 1.0, Y: 0.53}, {X: 1.0, Y: 0.62},
			{X: 0.75, Y: 0.56}, {X: 0.50, Y: 0.59},
			{X: 0.25, Y: 0.51}, {X: 0.0, Y: 0.54},
		},
	},
	"newyork": {
		{
			{X: 0.10, Y: 0.0}, {X: 0.17, Y: 0.0},
			{X: 0.20, Y: 0.35}, {X: 0.17, Y: 1.0},
			{X: 0.10, Y: 1.0},
		},
		{
			{X: 0.55, Y: 0.0}, {X: 0.62, Y: 0.0},
			{X: 0.64, Y: 0.45}, {X: 0.60, Y: 1.0},
			{X: 0.53, Y: 1.0},
		},
	},
	"tokyo": {
		{
			{X: 0.58, Y: 0.62}, {X: 1.0, Y: 0.52},
			{X: 1.0, Y: 1.0}, {X: 0.48, Y: 1.0},
		},
	},
}

type StationType string

const (
	Circle   StationType = "circle"
	Triangle StationType = "triangle"
	Square   StationType = "square"
	Pentagon StationType = "pentagon"
	Diamond  StationType = "diamond"
	Star     StationType = "star"
	Cross    StationType = "cross"
)

func AllTypes() []StationType {
	return []StationType{Circle, Triangle, Square, Pentagon, Diamond, Star, Cross}
}

func BasicTypes() []StationType {
	return []StationType{Circle, Triangle, Square}
}

func SpecialTypes() []StationType {
	return []StationType{Pentagon, Diamond, Star, Cross}
}
