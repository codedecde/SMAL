
#!/bin/bash

# usage: bash get_embeddings.sh --type=glove --dim=50
# usage: bash get_embeddings.sh --type=elmo
# usage: bash get_embeddings.sh --type=fasttext --lang=[en|nl|es|de]

# Default arguments
EMBED_DIM=50
EMBED_TYPE=glove
EMBED_LANG=en
EMBED_BASE_DIR=data/embeddings

# Argument parsing
for i in "$@"
do
case $i in
	-t=*|--type=*)
	EMBED_TYPE="${i#*=}"
	shift
	;;
	-d=*|--dim=*)
	EMBED_DIM="${i#*=}"
	shift
	;;
	-l=*|--lang=*)
	EMBED_LANG="${i#*=}"
	shift
	;;
	*)
	;;
esac
done

# Function for creating the embedding folder
function make_embed_dir () {
	local embed_dir=$1
	if [ ! -d ${embed_dir} ]; then
		mkdir -p ${embed_dir}
		touch ${embed_dir}/.gitkeep
	fi
}
# Get embeddings based on parameters passed
if [ ${EMBED_TYPE} = "glove" ]; then
	if [ ${EMBED_LANG} = "en" ]; then
		if [ ${EMBED_DIM} = "" ]; then
			EMBED_DIM=50
		fi
		EMBED_FILE=glove.6B.${EMBED_DIM}d.txt.gz
		EMBED_SRC=https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/${EMBED_FILE}
		EMBED_DIR=${EMBED_BASE_DIR}/glove
		make_embed_dir ${EMBED_DIR}
		DEST_EMBED_PATH=${EMBED_DIR}/${EMBED_FILE}
		echo "${DEST_EMBED_PATH}"
		if [ ! -f ${DEST_EMBED_PATH} ]; then
			echo "Downloading ${EMBED_FILE}"
			wget -O ${DEST_EMBED_PATH} ${EMBED_SRC}
		fi
	elif [ ${EMBED_LANG} = "de" ]; then
		EMBED_DIM=300
		EMBED_FILE=glove.de.${EMBED_DIM}d.txt.gz
		EMBED_TEXT_FILE=glove.de.${EMBED_DIM}d.txt
		EMBED_SRC=https://int-emb-glove-de-wiki.s3.eu-central-1.amazonaws.com/vectors.txt
		EMBED_DIR=${EMBED_BASE_DIR}/glove
		make_embed_dir ${EMBED_DIR}
		DEST_EMBED_PATH=${EMBED_DIR}/${EMBED_FILE}
		DEST_EMBED_TEXT_PATH=${EMBED_DIR}/${EMBED_TEXT_FILE}
		echo "${DEST_EMBED_PATH}"
		if [ ! -f ${DEST_EMBED_PATH} ]; then
			if [ ! -f ${DEST_EMBED_TEXT_PATH} ]; then
				echo "Downloading ${DEST_EMBED_TEXT_PATH}"
				wget -O ${DEST_EMBED_TEXT_PATH} ${EMBED_SRC}
			fi
			echo "Zipping the embeddings"
			gzip -c ${DEST_EMBED_TEXT_PATH} > ${DEST_EMBED_PATH}
		fi
	else
		echo "${EMBED_LANG} embeddings not supported"
	fi
elif [ ${EMBED_TYPE} = "fasttext" ]; then
	EMBED_DIR=${EMBED_BASE_DIR}/fasttext
	make_embed_dir ${EMBED_DIR}
	EMBED_DIR=${EMBED_DIR}/${EMBED_LANG}
	make_embed_dir ${EMBED_DIR}
	EMBED_FILE="cc."${EMBED_LANG}".300.vec.gz"
	SRC_URL="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl"
	DEST_EMBED_PATH=${EMBED_DIR}/${EMBED_FILE}
	EMBED_SRC=${SRC_URL}/${EMBED_FILE}
	echo "${DEST_EMBED_PATH}"
	if [ ! -f ${DEST_EMBED_PATH} ]; then
		echo "Downloading ${EMBED_FILE}"
		wget -O ${DEST_EMBED_PATH} ${EMBED_SRC}
	fi
elif [ ${EMBED_TYPE} = "elmo" ]; then
	OPT_SRC=https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
	WEIGHTS_SRC=https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
	EMBED_DIR=${EMBED_BASE_DIR}/elmo
	make_embed_dir ${EMBED_DIR}
	OPT_FILE=${EMBED_DIR}/"elmo_2x4096_512_2048cnn_2xhighway_options.json"
	WEIGHTS_FILE=${EMBED_DIR}/"elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
	if [ ! -f ${OPT_FILE} ]; then
		echo "Downloading ${OPT_FILE}"
		wget -O ${OPT_FILE} ${OPT_SRC}
	fi
	if [ ! -f ${WEIGHTS_FILE} ]; then
		echo "Downloading ${WEIGHTS_FILE}"
		wget -O ${WEIGHTS_FILE} ${WEIGHTS_SRC}
	fi
elif [ ${EMBED_TYPE} = "bert" ]; then
	BERT_VOCAB="https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt"
	BERT_MODEL="https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz"
	MODEL_NAME="bert-base-cased"
	EMBED_DIR=${EMBED_BASE_DIR}/bert/${MODEL_NAME}
	make_embed_dir ${EMBED_DIR}
	VOCAB_FILE=${EMBED_DIR}/${BERT_VOCAB}
	MODEL_FILE=${EMBED_DIR}/${BERT_MODEL}
	VOCAB_OUTPUT_FILE=${EMBED_DIR}/vocab.txt
	MODEL_OUTPUT_FILE=${EMBED_DIR}/${MODEL_NAME}.tar.gz
	if [ ! -f ${VOCAB_OUTPUT_FILE} ]; then
		echo "Downloading ${VOCAB_OUTPUT_FILE}"
		wget -O ${VOCAB_OUTPUT_FILE} ${BERT_VOCAB}
	fi
	if [ ! -f ${MODEL_OUTPUT_FILE} ]; then
		echo "Downloading ${MODEL_OUTPUT_FILE}"
		wget -O ${MODEL_OUTPUT_FILE} ${BERT_MODEL}
	fi
elif [ ${EMBED_TYPE} = "mbert" ]; then
	BERT_VOCAB="https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt"
	BERT_MODEL="https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz"
	MODEL_NAME="bert-base-multilingual-cased"
	EMBED_DIR=${EMBED_BASE_DIR}/mbert/${MODEL_NAME}
	make_embed_dir ${EMBED_DIR}
	VOCAB_FILE=${EMBED_DIR}/${BERT_VOCAB}
	MODEL_FILE=${EMBED_DIR}/${BERT_MODEL}
	VOCAB_OUTPUT_FILE=${EMBED_DIR}/vocab.txt
	MODEL_OUTPUT_FILE=${EMBED_DIR}/${MODEL_NAME}.tar.gz
	if [ ! -f ${VOCAB_OUTPUT_FILE} ]; then
		echo "Downloading ${VOCAB_OUTPUT_FILE}"
		wget -O ${VOCAB_OUTPUT_FILE} ${BERT_VOCAB}
	fi
	if [ ! -f ${MODEL_OUTPUT_FILE} ]; then
		echo "Downloading ${MODEL_OUTPUT_FILE}"
		wget -O ${MODEL_OUTPUT_FILE} ${BERT_MODEL}
	fi
else
	echo "TYPE ${EMBED_TYPE} unrecognized"
fi
