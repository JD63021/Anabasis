# Integrate this into your existing `~/fvm-cuda` tree

Copy these folders into the root of your existing project:

- `libpoisson/`
- `apps/poisson_mms_lib/`

Then add these two lines to your existing root `CMakeLists.txt`:

```cmake
add_subdirectory(libpoisson)
add_subdirectory(apps/poisson_mms_lib)
```

If you prefer, you can also build this package standalone from this folder.
