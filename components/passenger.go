package components

import "minimetro-go/config"

type Passenger struct {
	CurrentStation       *Station
	Destination          config.StationType
	OnTrain              *Train
	Path                 []*Station
	PathIndex            int
	WaitStartTime        float64
	LastRouteCalculation float64
	ReservedTrain        *Train
}

func NewPassenger(station *Station, destination config.StationType, currentTime float64) *Passenger {
	return &Passenger{
		CurrentStation:       station,
		Destination:          destination,
		WaitStartTime:        currentTime,
		LastRouteCalculation: currentTime,
	}
}
