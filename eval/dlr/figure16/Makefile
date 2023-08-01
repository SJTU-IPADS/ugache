.PHONY: run plot
run:
	python runner.py
plot:
	python parser.py
	@read -r refresh_start refresh_stop < data_refresh.dat; \
	gnuplot -e "refresh_start=$$refresh_start; refresh_stop=$$refresh_stop" plot.plt
plot-paper:
	python parser.py run-logs-paper
	@read -r refresh_start refresh_stop < data_refresh.dat; \
	gnuplot -e "refresh_start=$$refresh_start; refresh_stop=$$refresh_stop" plot.plt
