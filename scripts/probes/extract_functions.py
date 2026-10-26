import sys

import coremltools as ct

src, out, fns = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
d = ct.utils.MultiFunctionDescriptor()
for fn in fns:
    d.add_function(src, src_function_name=fn, target_function_name=fn)
d.default_function_name = fns[0]
ct.utils.save_multifunction(d, out)
print("saved", out)
