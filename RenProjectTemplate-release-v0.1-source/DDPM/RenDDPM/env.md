# How to config `env.json`

An example for `/RenDDPM/env.json`.

For "h5dataset", You can fill relative or absolute path. It will first check existance as absolute, and then fall back to relative.

```json
{
"result_folder_name":"Results",
"temp_folder_name":"Temp",
"num_workers":16,
"animation.ffmpeg_path":"/path/to/your/ffmpeg"
}

```

Use `which ffmpeg` to find the path, or in win, the path to ffmpeg.exe

