#!/usr/bin/env python3
import sys
from io import StringIO
from tensorflow.keras.models import load_model

output = StringIO()

try:
    print("Loading model...", file=output)
    m = load_model('flower_model_final.keras')
    
    print("\n" + "="*80, file=output)
    print("MODEL SUMMARY", file=output)
    print("="*80, file=output)
    
    m.summary(print_fn=lambda x: print(x, file=output))
    
    print("\n" + "="*80, file=output)
    print("MODEL DETAILS", file=output)
    print("="*80, file=output)
    print(f"Input shape: {m.input_shape}", file=output)
    print(f"Output shape: {m.output_shape}", file=output)
    print(f"Model type: {type(m).__name__}", file=output)
    print(f"Number of layers: {len(m.layers)}", file=output)
    
    print("\n" + "="*80, file=output)
    print("ALL LAYERS", file=output)
    print("="*80, file=output)
    for i, layer in enumerate(m.layers):
        print(f"{i}: {layer.name:30} - {type(layer).__name__:20} - {str(layer.output_shape)}", file=output)
    
    # Write to file
    with open('model_summary.txt', 'w', encoding='utf-8') as f:
        f.write(output.getvalue())
    
    print(output.getvalue())

except Exception as e:
    print(f"Error: {e}", file=output)
    import traceback
    traceback.print_exc(file=output)
    with open('model_summary.txt', 'w', encoding='utf-8') as f:
        f.write(output.getvalue())
