# MiraeCity
**MILab 미래시티 과제 repo**

정재준 개인 공부 정리 브랜치

## Kaldi Install menual 

git clone <https://github.com/kaldi-asr/kaldi.git> kaldi --origin upstream
1) 외부 라이브러리 설치
cd kaldi/tools
./extras/check_dependencies.sh 

필요한 패키지가 설치가 되어있지 않으면 여기서 뭐뭐 설치하라고 알려줍니다.
make -j N_CPUS

2) Kaldi source code compile
cd ../src

CUDA 설치가 되어 있으시다면,
./configure --shared

CUDA 설치가 안되어있고 GPU를 안(못)쓰실거면,
./configure --use-cuda=no --shared
make depend -j N_CPUS
make -j N_CPUS

### TIL
230410 : sample dataset added
230413 : Training librispeech with deepspeech; working on Korean dataset preprocessing
230415 : Still training trying to make demo 
230430 : RF training
### ref
- https://medium.com/@indra622/0-kaldi-asr-tutorial-b036b6dac26e - 칼디 install tutorial
- https://medium.com/@indra622/1-kaldi-directory를-통한-전체-구조-설명-bb9107433945 - 칼디 architecture
- https://hub.docker.com/r/quleyuan9826/deepspeech2-cuda9.0-pytorch - deepspeech2 docker image
- https://todayisbetterthanyesterday.tistory.com/51 - RF sample
