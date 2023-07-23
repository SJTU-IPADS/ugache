.PHONY: run plot
run:
	python scripts-ugache/runner.py
	python scripts-gnnlab/runner.py
plot:
	python scripts-ugache/parser.py
	python scripts-gnnlab/parser.py
	gnuplot plot.plt
plot-paper:
	python scripts-ugache/parser.py run-logs-paper
	python scripts-gnnlab/parser.py run-logs-paper
	gnuplot plot.plt
