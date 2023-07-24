.PHONY: run plot
run:
	python runner.py
plot:
	python parser.py
	gnuplot plot.plt
plot-paper:
	python parser.py run-logs-paper
	gnuplot plot.plt
