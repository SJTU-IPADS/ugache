#!/usr/bin/env gnuplot

dat_file="data.dat"
set datafile sep '\t'
set output "data.eps"

col_app                          = 1
col_cache_policy                 = 2
col_dataset                      = 3
col_step_time_feat_copy          = 4

set terminal postscript "Helvetica,16" eps enhance color dl 2 background "white"

set pointsize 1
set size 0.6,0.5
set tics font ",14" scale 0.5

set style data histogram
set style histogram clustered gap 1
set style fill solid border -1
set boxwidth 0.6 relative

format_str="<awk -F'\\t' '{ if(". \
                              "$".col_app."           ~ \"%s\"     && ". \
                              "$".col_dataset."       ~ \"%s\"     && ". \
                              "$".col_cache_policy."  ~ \"%s\"     ". \
                              ") { print }}' %s "
cmd_filter_dat_by_policy(file,app,dataset,policy, percent)=sprintf(format_str, app, dataset, policy, file)

fs_pattern_A = 3
fs_pattern_B = 3
fs_pattern_C = 3

set label "{/Helvetica-Italic Server C}" at graph 0.05, graph 0.9 textcolor rgb "#000000"
set border lc rgb "#000000" lw 2 
set tics textcolor "black"
set grid y
unset yrange; set yrange [0:]
set xlabel "DLRM" offset 0,0.3
set ylabel "Embedding Extraction Time(ms)" offset 2.,0 font ',13' 
set xtics nomirror offset 0,0.3 textcolor rgb "#000000"
set ytics offset 0.5,0 #format "%.1f" #nomirror
set key outside right Right top enhanced nobox # autotitles columnhead
set key samplen 1 spacing 1 height 5 width 0 font ',13' 
set arrow from  graph 0.2, graph -0.12 to graph 0.8, graph -0.12 nohead lt 1 lw 2 lc "#000000" front
plot \
     cmd_filter_dat_by_policy(dat_file, "dlrm", ".*", "^HPS*",               ".*") using (1000*column(col_step_time_feat_copy)):xticlabels(col_dataset) w histogram fs pattern 2 lc rgb "#af2318" t "HPS", \
     cmd_filter_dat_by_policy(dat_file, "dlrm", ".*", "^SOK*",               ".*") using (1000*column(col_step_time_feat_copy)):xticlabels(col_dataset) w histogram fs pattern 2 lc rgb "#f19e38" t "SOK", \
     cmd_filter_dat_by_policy(dat_file, "dlrm", ".*", "^Rep*",               ".*") using (1000*column(col_step_time_feat_copy)):xticlabels(col_dataset) w histogram fs pattern fs_pattern_A lc rgb "#af2318" t "Rep", \
     cmd_filter_dat_by_policy(dat_file, "dlrm", ".*", "^Cliq*",              ".*") using (1000*column(col_step_time_feat_copy)):xticlabels(col_dataset) w histogram fs pattern fs_pattern_A lc rgb "#f19e38" t "Part", \
     cmd_filter_dat_by_policy(dat_file, "dlrm", ".*", "Coll",                ".*") using (1000*column(col_step_time_feat_copy)):xticlabels(col_dataset) w histogram fs pattern fs_pattern_A lc rgb "#0080ff" t "UGache", \

