#!/bin/bash
/home/develop/disks/sdb/Songzq/mosesdecoder/scripts/generic/multi-bleu.perl \
-lc /home/develop/disks/sdb/Songzq/emotional_chat_machine/data/stc_data/train_test/test.response.filter.txt \
< /home/develop/disks/sdb/Songzq/emotional_chat_machine/exp/test/response.e200.lr0.1.1e-1.txt 