package summation

import "sync"

type summator struct {
	sum int
	mu  sync.Mutex
}

func (s *summator) add(n int) {
	s.mu.Lock()
	s.sum += n
	s.mu.Unlock()
}

func SumMutex(nums []int, concurrency int) int {
	s := &summator{}

	chunks := chunk(nums, (len(nums)+concurrency-1)/concurrency)

	var wg sync.WaitGroup
	for _, chunk := range chunks {
		wg.Add(1)

		go func(chunk []int) {
			defer wg.Done()
			s.add(sum(chunk))
		}(chunk)
	}

	wg.Wait()

	return s.sum
}