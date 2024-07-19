#/bin/bash

cp workflows/tess_injection_recovery/_figures/searched/1019692.pdf latex/figures/searched_1019692.pdf
cp workflows/tess_injection_recovery/_figures/cleaned/1019692.pdf latex/figures/cleaned_1019692.pdf
find workflows -type d -maxdepth 2 -name 'figures' -exec sh -c 'cp -r "$0"/* latex/figures' "{}" \;
cp -r "latex/figures/TOI_540" "latex/figures/TOI 540"