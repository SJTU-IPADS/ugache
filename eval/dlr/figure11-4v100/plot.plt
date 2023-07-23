#!/usr/bin/env gnuplot

dat_file="data.dat"
set datafile sep '\t'
set output "data.eps"

col_app                          = 1
col_cache_policy                 = 2
col_dataset                      = 3
col_step_time_train_total        = 4

set terminal postscript "Helvetica,16" eps enhance color dl 2 background "white"

set pointsize 1
set size 0.7,0.35

set tics font ",14" scale 0.5

set style data histogram
set style histogram clustered gap 1
set style fill solid border -1
set boxwidth 0.75 relative

format_str="<awk -F'\\t' '{ if(". \
                              "$".col_app."           ~ \"%s\"     && ". \
                              "$".col_dataset."       ~ \"%s\"     && ". \
                              "$".col_cache_policy."  ~ \"%s\"     ". \
                              ") { print }}' %s "
cmd_filter_dat_by_policy(file,app,dataset,policy, percent)=sprintf(format_str, app, dataset, policy, file)

### Key
set key inside left Right top enhanced nobox # autotitles columnhead
set key samplen 1.5 spacing 1.5 height 0.2 width 0 font ',13' center at graph 0.1, graph 0.8 #maxrows 1 at graph 0.02, graph 0.975  noopaque
unset key

set multiplot layout 1,2

## Y-axis
set yrange [0:]
set ytics offset 0.5,0 #format "%.1f" #nomirror
set grid y

set xtics nomirror offset 0,0.3

fs_pattern_A = 3
fs_pattern_B = 3
fs_pattern_C = 3

handle_multiplot_location(row, col) = sprintf("set size 0.35, 0.33; set origin %d*0.35, 0.3-(%d+1)*0.3;", col,row)

set xrange[-0.75:1.75]

set label "{/Helvetica-Italic Server A}" at graph 0.05, graph 0.9 textcolor rgb "#006000"
set border lc rgb "#006000" lw 2
set tics textcolor "black"
eval(handle_multiplot_location(0,0))
set key outside left Right top enhanced nobox # autotitles columnhead
set key samplen 4 spacing 0 height 3 width 4 font ',16' maxrows 1 center at graph 1, graph 1.1 #maxrows 1 at graph 0.02, graph 0.975  noopaque
unset xtics
set yrange [0:20]
plot cmd_filter_dat_by_policy(dat_file, "dlrm", ".*", "^HPS*",         ".*") using (1000*column(col_step_time_train_total)):xticlabels(col_dataset) w histogram fs pattern fs_pattern_A lc rgb "#af2318" t "HPS", \
     cmd_filter_dat_by_policy(dat_file, "dlrm", ".*", "^SOK*",         ".*") using (1000*column(col_step_time_train_total)):xticlabels(col_dataset) w histogram fs pattern fs_pattern_A lc rgb "#f19e38" t "SOK", \
     cmd_filter_dat_by_policy(dat_file, "dlrm", ".*", "UGache*", ".*") using (1000*column(col_step_time_train_total)):xticlabels(col_dataset) w histogram fs pattern fs_pattern_A lc rgb "#0080ff" t "UGache"

unset key
unset label

eval(handle_multiplot_location(0,1))
set yrange [0:50]
plot cmd_filter_dat_by_policy(dat_file, "dcn", ".*", "^HPS*",          ".*") using (1000*column(col_step_time_train_total)):xticlabels(col_dataset) w histogram fs pattern fs_pattern_A lc rgb "#af2318" t "HPS", \
     cmd_filter_dat_by_policy(dat_file, "dcn", ".*", "^SOK*",          ".*") using (1000*column(col_step_time_train_total)):xticlabels(col_dataset) w histogram fs pattern fs_pattern_A lc rgb "#f19e38" t "SOK", \
     cmd_filter_dat_by_policy(dat_file, "dcn", ".*", "UGache*",  ".*") using (1000*column(col_step_time_train_total)):xticlabels(col_dataset) w histogram fs pattern fs_pattern_A lc rgb "#0080ff" t "UGache"

unset key