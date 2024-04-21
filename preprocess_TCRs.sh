awk '{gsub(" ", "", $0); gsub("\\.", "", $0); print}' "$1" > temp_file && mv temp_file "$1"
