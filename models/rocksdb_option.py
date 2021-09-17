import random

KB = 1024
MB = 1024 * 1024


# Change parameter values from line 45 to line 88
option = {
    "max_background_compactions": [i for i in range(1, 17)], # D:1, B:4 ~ 32
    "max_background_flushes": [i for i in range(1, 17)], #D:1, B:4~32
    "write_buffer_size": [s * KB for s in range(512, 2048)], #D:64M, B:0.25M ~ 1M
    "max_write_buffer_number": [i for i in range(2, 9)], #D:2, B:2~16
    "min_write_buffer_number_to_merge": [i for i in range(1, 3)], #D:1
    "compaction_pri": { #D:0
        "kByCompensatedSize" : 0,
        "kOldestLargestSeqFirst" : 1,
        "kOldestSmallestSeqFirst" : 2,
        "kMinOverlappingRatio" : 3
    },
    "compaction_style": { #D:0
        "kCompactionStyleLevel" : 0, 
        "kCompactionStyleUniversal" : 1,
        "kCompactionStyleFIFO" : 2,
        "kCompactionStyleNone" : 3
    },
    "level0_file_num_compaction_trigger": [i for i in range(2, 9)], #D:4, B:2 ~ 8
    "level0_slowdown_writes_trigger": [i for i in range(16, 33)], #D:20, B:16 ~ 64
    "level0_stop_writes_trigger": [i for i in range(32, 65)], #D:36, B:64 ~ 128
    "compression_type": [i for i in range(0, 4)], #D:"snappy", B:no "bzip2"
    "bloom_locality": [0, 1], #D:0
    "open_files": [-1, 10000, 100000, 1000000], #D:-1 B:-1
    "block_size": [s * KB for s in range(2, 17)], #D:4096, B:4096 ~ 32768
    "cache_index_and_filter_blocks": [1, 0] #D:false
}

plus_option = {
    "memtable_bloom_size_ratio": [0, 0.05, 0.1, 0.15, 0.2], #D:0
    "compression_ratio": [i/100 for i in range(100)] #D:0.5, B:0.1 ~ 0.9
}

level_compaction_option = {
    "max_bytes_for_level_base": [s * MB for s in range(2, 9)], #D:256M, B:1M ~ 16M
    "max_bytes_for_level_multiplier": [i for i in range(8, 13)], #D:10, B:6 ~ 10
    "target_file_size_base": [s * KB for s in range(512, 2049)], #D:64M, B:0.25M ~ 4M
    "target_file_size_multiplier": [ i for i in range(1, 3)], #D:1, B:1 ~ 2
    "num_levels": [5, 6, 7, 8] #D:7, B:7
}

universal_compaction_option = {
    "universal_max_size_amplification_percent": [],
    "universal_size_ratio ": [],
    "universal_min_merge_width": [],
    "universal_max_size_amplification_percent": [],
    "universal_compression_size_percent": []
}

def make_random_option():
    option_list = ""
    option_dict = {}

    compaction_style = ""
    write_buffer_number = -1

    compression_type = 0

    # option
    for k, v in option.items():

        if k == "max_write_buffer_number":
            write_buffer_number = random.choice(v)
            value = write_buffer_number
            opt = f"-{k}={value} "
        elif k == "min_write_buffer_number_to_merge":
            lt_write_buffers = [i for i in v if i <= write_buffer_number]
            value = random.choice(lt_write_buffers)
            opt = f"-{k}={value} "
        elif k == "compaction_pri":
            value = random.choice(list(v.values()))
            opt = f"-{k}={value} "
        elif k == "compaction_style":
            # compaction_style = random.choice(list(v.keys()))
            compaction_style = "kCompactionStyleLevel"
            value = v[compaction_style]
            opt = f"-{k}={value} "
        elif k == "compression_type" and v == "none":
            compression_type = 1 # to make compression ratio be 1
        else:
            value = random.choice(v)
            opt = f"-{k}={value} "

        option_list += opt
        option_dict[k] = value
    
    
    # compaction option
    if compaction_style == "kCompactionStyleLevel":
        for k, v in level_compaction_option.items():
            value = random.choice(v)
            opt = f"-{k}={value} "
            option_list += opt
            option_dict[k] = value  
    elif compaction_style == "kCompactionStyleUniversal":
        for k, v in universal_compaction_option.items():
            value = random.choice(v)
            opt = f"-{k}={value} "
            option_list += opt
            option_dict[k] = value
    else:
        pass


    # plus_option
    for k, v in plus_option.items():
        if k == "compression_ratio" and compression_type:
            value = 1
        else:
            value = random.choice(v)
        opt = f"-{k}={value} "
        option_list += opt
        option_dict[k] = value
    
    return (option_list, option_dict)