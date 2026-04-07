package state

import (
	"minimetro-go/components"
	"slices"
)

type GameState struct {
	Paused                bool
	SpawnStationsEnabled  bool
	Speed                 float64
	FastForward           int // 0=1x, 1=2x, 2=4x
	Score                 int
	Week                  int
	Day                   int
	Stations              []*components.Station
	Lines                 []*components.Line
	Trains                []*components.Train
	Passengers            []*components.Passenger
	SelectedLine          int
	MaxLines              int
	AvailableLines        int
	Bridges               int
	Carriages             int
	Interchanges          int
	AvailableTrains       int
	GameOver              bool
	SelectedCity          string
	Rivers                []*components.River
	LastSpawnTime         float64
	LastStationSpawnTime  float64
	WeekStartTime         float64
	StationIDCounter      int
	TrainIDCounter        int
	PassengersDelivered   int
	GameStartTime         float64
	CameraZoom            float64
	SimTimeMs             float64
	GraphDirty            bool
}

// AddStation appends station to the game and increments the ID counter.
func (gs *GameState) AddStation(s *components.Station) {
	gs.Stations = append(gs.Stations, s)
	gs.StationIDCounter++
}

// AddTrain appends train to the global train list and increments the ID counter.
func (gs *GameState) AddTrain(t *components.Train) {
	gs.Trains = append(gs.Trains, t)
	gs.TrainIDCounter++
}

// RemoveTrain removes train from the global train list. No-op if not present.
func (gs *GameState) RemoveTrain(t *components.Train) {
	if i := slices.Index(gs.Trains, t); i >= 0 {
		gs.Trains = slices.Delete(gs.Trains, i, i+1)
	}
}

// AddPassenger appends passenger to the global passenger list.
func (gs *GameState) AddPassenger(p *components.Passenger) {
	gs.Passengers = append(gs.Passengers, p)
}

// RemovePassenger removes passenger from the global passenger list. No-op if not present.
func (gs *GameState) RemovePassenger(p *components.Passenger) {
	if i := slices.Index(gs.Passengers, p); i >= 0 {
		gs.Passengers = slices.Delete(gs.Passengers, i, i+1)
	}
}

func NewGameState() *GameState {
	gs := &GameState{}
	gs.Reset()
	return gs
}

func (gs *GameState) Reset() {
	gs.Paused = false
	gs.SpawnStationsEnabled = true
	gs.Speed = 1.0
	gs.FastForward = 0
	gs.Score = 0
	gs.Week = 1
	gs.Day = 0
	gs.Stations = make([]*components.Station, 0)
	gs.Lines = make([]*components.Line, 0)
	gs.Trains = make([]*components.Train, 0)
	gs.Passengers = make([]*components.Passenger, 0)
	gs.SelectedLine = 0
	gs.MaxLines = 5
	gs.Bridges = 2
	gs.Carriages = 0
	gs.Interchanges = 0
	gs.AvailableTrains = 3
	gs.GameOver = false
	gs.SelectedCity = "london"
	gs.Rivers = make([]*components.River, 0)
	gs.StationIDCounter = 0
	gs.TrainIDCounter = 0
	gs.PassengersDelivered = 0
	gs.CameraZoom = 1.0
	gs.SimTimeMs = 0.0
	gs.GraphDirty = true
}
