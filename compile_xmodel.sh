
#!/bin/bash
# Shield-Ryzen V2 — AOT Compilation for AMD XDNA
# Requires: Vitis AI 3.5+ installed (vai_c_xir)
#
# Usage: ./compile_xmodel.sh [xdna1|xdna2]
#
# Outputs: models/compiled/[target]/*.xmodel

TARGET=$1
if [ -z "$TARGET" ]; then
    echo "Usage: ./compile_xmodel.sh [xdna1|xdna2]"
    echo "Defaulting to xdna1"
    TARGET="xdna1"
fi

echo "🚀 Compiling models for AMD XDNA ($TARGET)..."

# Ensure output directory exists
OUT_DIR="models/compiled/$TARGET"
mkdir -p "$OUT_DIR"

# 1. Compile XceptionNet INT8
if [ -f "shield_ryzen_int8.onnx" ]; then
    echo "Compiling ShieldXception..."
    vai_c_xir \
        --xmodel shield_ryzen_int8.onnx \
        --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/Alveo/arch.json \
        --output_dir "$OUT_DIR" \
        --net_name shield_xception
else
    echo "⚠️ shield_ryzen_int8.onnx not found. Skipping."
fi

# 2. Compile BlazeFace (Face Detection) if available
if [ -f "models/blazeface_int8.onnx" ]; then
    echo "Compiling BlazeFace..."
    vai_c_xir \
        --xmodel models/blazeface_int8.onnx \
        --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/Alveo/arch.json \
        --output_dir "$OUT_DIR" \
        --net_name blazeface
fi

# 3. Compile Attribution Classifier if available
if [ -f "models/attribution_classifier.onnx" ]; then
    echo "Compiling Attribution Classifier..."
    vai_c_xir \
        --xmodel models/attribution_classifier.onnx \
        --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/Alveo/arch.json \
        --output_dir "$OUT_DIR" \
        --net_name attribution
fi

echo "✅ Compilation attempts finished. Check $OUT_DIR for .xmodel files."
