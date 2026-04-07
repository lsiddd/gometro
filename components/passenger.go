package components

type Passenger struct {
	CurrentStation       *Station
	Destination          string
	OnTrain              *Train
	Path                 []*Station
	PathIndex            int
	WaitStartTime        float64
	LastRouteCalculation float64
	ReservedTrain        *Train
}

func NewPassenger(station *Station, destination string, currentTime float64) *Passenger {
	return &Passenger{
		CurrentStation:       station,
		Destination:          destination,
		WaitStartTime:        currentTime,
		LastRouteCalculation: currentTime,
	}
}
