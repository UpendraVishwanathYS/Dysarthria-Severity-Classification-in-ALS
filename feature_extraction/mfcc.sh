find ./data -iname "*.wav" > files
sed -e 's/.wav//g;s/data//g;s/\///g;s/\.//g' files > unique
paste unique files | sed 's/\t/ /g' > wav.scp
#Path
path=/home/wtc8/kaldi/src/featbin
mkdir mfcc_norm_txt;
count=`ls data | wc -l`

for i in `seq 1 $count`;
do
cat wav.scp | head -n $i | tail -1 > wav_1.scp
        name=`cat wav_1.scp | cut -d " "  -f1`
        name=$name'.txt';
#Computing mfccs and copying to archive
$path/compute-mfcc-feats --frame-length=20 --frame-shift=10 scp,p:wav_1.scp ark:- | $path/copy-feats --compress=true ark:- ark,scp:test.ark,feats.scp
$path/add-deltas ark:test.ark ark,scp:test_delta.ark,feats_delta.scp
$path/compute-cmvn-stats scp:feats_delta.scp ark,scp:cmvn.ark,cmvn.scp
$path/apply-cmvn scp:cmvn.scp scp:feats_delta.scp ark,t:mfcc_norm_txt/$name
sed -i '1d' mfcc_norm_txt/$name
sed -i "s/\]//g" mfcc_norm_txt/$name
done

rm -r files unique wav.scp wav_1.scp feats.scp test.ark test_delta.ark feats_delta.scp cmvn.scp cmvn.ark
