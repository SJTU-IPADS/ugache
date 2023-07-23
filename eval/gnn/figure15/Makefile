.PHONY: run plot
run:
	@echo "skipping run figure15 since it shares log with figure14"
plot:
	python parser.py ../figure14/run-logs
	gnuplot plot.plt
plot-paper:
	python parser.py ../figure14/run-logs-paper
	gnuplot plot.plt
