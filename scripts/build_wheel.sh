#!/usr/bin/env bash

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#=================================================
#                   Utils
#=================================================


# directory config
DIST_DIR="dist"
BUILD_DIR="build"
EGG_DIR="paddlemix.egg-info"

# command line log config
RED='\033[0;31m'
BLUE='\033[0;34m'
GREEN='\033[1;32m'
BOLD='\033[1m'
NONE='\033[0m'

function python_version_check() {
  PY_MAIN_VERSION=`python -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $1}'`
  PY_SUB_VERSION=`python -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $2}'`
  echo -e "find python version ${PY_MAIN_VERSION}.${PY_SUB_VERSION}"
  if [ $PY_MAIN_VERSION -ne "3" -o $PY_SUB_VERSION -lt "5" ]; then
    echo -e "${RED}FAIL:${NONE} please use Python >= 3.5 !"
    exit 1
  fi
}

function init() {
    echo -e "${BLUE}[init]${NONE} removing building directory..."
    rm -rf $DIST_DIR $BUILD_DIR $EGG_DIR
    if [ `pip list | grep paddlemix | wc -l` -gt 0  ]; then
      echo -e "${BLUE}[init]${NONE} uninstalling paddlemix..."
      pip uninstall -y paddlemix
    fi
    echo -e "${BLUE}[init]${NONE} ${GREEN}init success\n"
}

function build_and_install() {
  echo -e "${BLUE}[build]${NONE} building paddlemix wheel..."
  # add ppdiffusers as dependency to paddlemix
  cp requirements.txt requirements.bak
  echo 'ppdiffusers==0.19.3' >> requirements.txt 
  python setup.py sdist bdist_wheel
  if [ $? -ne 0 ]; then
    echo -e "${RED}[FAIL]${NONE} build paddlemix wheel failed !"
    exit 1
  fi
  echo -e "${BLUE}[build]${NONE} ${GREEN}build paddldet wheel success\n"
  mv requirements.bak requirements.txt

  echo -e "${BLUE}[install]${NONE} installing paddlemix..."
  cd $DIST_DIR

  find . -name "paddlemix*.whl" | xargs pip install
  if [ $? -ne 0 ]; then
    cd ..
    echo -e "${RED}[FAIL]${NONE} install paddlemix wheel failed !"
    exit 1
  fi
  echo -e "${BLUE}[install]${NONE} ${GREEN}paddlemix install success\n"
  cd ..
}

function unittest() {
  echo -e "${BLUE}[unittest]${NONE} run unittests..."

  # NOTE: perform unittests make sure installed paddlemix is used
  python -m unittest discover -v 

  echo -e "${BLUE}[unittest]${NONE} ${GREEN}unittests success\n${NONE}"
}

function cleanup() {
  rm -rf $BUILD_DIR $EGG_DIR
  pip uninstall -y paddlemix
}

function abort() {
  echo -e "${RED}[FAIL]${NONE} build wheel and unittest failed !
          please check your code" 1>&2

  cur_dir=`basename "$pwd"`
  if [ cur_dir==$TEST_DIR -o cur_dir==$DIST_DIR ]; then
    cd ..
  fi

  rm -rf $BUILD_DIR $EGG_DIR $DIST_DIR $TEST_DIR
  pip uninstall -y paddlemix
}

python_version_check

trap 'abort' 0
set -e

init
build_and_install
# unittest
cleanup

# get Paddle version
PADDLE_VERSION=`python -c "import paddle; print(paddle.version.full_version)"`
PADDLE_COMMIT=`python -c "import paddle; print(paddle.version.commit)"`
PADDLE_COMMIT=`git rev-parse --short $PADDLE_COMMIT`

# get PaddleMIX branch
PPDET_BRANCH=`git rev-parse --abbrev-ref HEAD`
PPDET_COMMIT=`git rev-parse --short HEAD`

# get Python version
PYTHON_VERSION=`python -c "import platform; print(platform.python_version())"`

echo -e "\n${GREEN}paddlemix wheel compiled and checked success !${NONE}
        ${BLUE}Python version:${NONE} $PYTHON_VERSION
        ${BLUE}Paddle version:${NONE} $PADDLE_VERSION ($PADDLE_COMMIT)
        ${BLUE}paddlemix branch:${NONE} $PPDET_BRANCH ($PPDET_COMMIT)\n"

echo -e "${GREEN}wheel saved under${NONE} ${RED}${BOLD}./dist"

trap : 0
