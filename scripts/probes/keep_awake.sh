#!/bin/zsh
# usage: keep_awake.sh [--keep-demo]   keeps the phone awake without touching any phone setting (Auto-Lock "Never" is not
# available): terminates the demo app (unless --keep-demo), then puts the keeper in front. The keeper is the probe built
# as com.delavalom.egcm.keeper without models; with no arguments it loads nothing and stays in front with the idle timer
# off. Build: in ios-load-probe, move ProbeModels aside, then
#   xcodebuild ... -derivedDataPath /tmp/keeper-dd PRODUCT_BUNDLE_IDENTIFIER=com.delavalom.egcm.keeper PRODUCT_NAME=Keeper build
set -u
U=${UDID:?set UDID}
if [ "${1:-}" != "--keep-demo" ]; then
  pid=$(xcrun devicectl device info processes --device $U 2>/dev/null | awk '/PhotoSearchDemo.app\/PhotoSearchDemo/{print $1; exit}')
  [ -n "$pid" ] && xcrun devicectl device process terminate --device $U --pid $pid >/dev/null 2>&1
fi
xcrun devicectl device process launch --device $U --terminate-existing com.delavalom.egcm.keeper -- -tag keeper >/dev/null 2>&1 \
  || { echo "KEEPER LAUNCH FAILED (phone locked?)"; exit 1; }
