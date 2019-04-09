#!/bin/bash
/home/develop/disks/sdb/Songzq/mosesdecoder/scripts/generic/multi-bleu.perl \
	-lc /home/develop/disks/sdb/Songzq/emotional_chat_machine/data/stc_data/train_test/test.response.filter.txt \
	< /home/develop/disks/sdb/Songzq/emotional_chat_machine/exp/test/generation.data.txt