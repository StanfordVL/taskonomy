declare -a arr=("curvature" "depth" "edge" "edge2d" "keypoint" "keypoint2d" "mist" "normal" "reshade" "segment25d" "segment2d")

for i in "${arr[@]}"
do
    img_info=$(identify -verbose ~/s3/2fycrku4FjW/$i/point_0_view_0_domain_$i.png )
    dest=./infos/$i.txt
    echo -e "$img_info" > "./info/$i.txt"
done



