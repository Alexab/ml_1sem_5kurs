# by БорискинМА (30.11.19), 3341506/90401

import pandas as pd

dataset_1 = pd.DataFrame(columns=["error_name"])
dataset_2 = pd.DataFrame(columns=["State"])
dataset_3 = pd.DataFrame(columns=["Tgid"])
dataset_4 = pd.DataFrame(columns=["Pid"])
dataset_5 = pd.DataFrame(columns=["PPid"])
dataset_6 = pd.DataFrame(columns=["TracerPid"])
dataset_7 = pd.DataFrame(columns=["Uid"])
dataset_8 = pd.DataFrame(columns=["Gid"])
dataset_9 = pd.DataFrame(columns=["FDSize"])
dataset_10 = pd.DataFrame(columns=["Groups"])
dataset_11 = pd.DataFrame(columns=["VmPeak"])
dataset_12 = pd.DataFrame(columns=["VmSize"])
dataset_13 = pd.DataFrame(columns=["VmLck"])
dataset_14 = pd.DataFrame(columns=["VmPin"])
dataset_15 = pd.DataFrame(columns=["VmHWM"])
dataset_16 = pd.DataFrame(columns=["VmRSS"])
dataset_17 = pd.DataFrame(columns=["VmData"])
dataset_18 = pd.DataFrame(columns=["VmStk"])
dataset_19 = pd.DataFrame(columns=["VmExe"])
dataset_20 = pd.DataFrame(columns=["VmLib"])
dataset_21 = pd.DataFrame(columns=["VmPTE"])
dataset_22 = pd.DataFrame(columns=["VmSwap"])
dataset_23 = pd.DataFrame(columns=["Threads"])
dataset_24 = pd.DataFrame(columns=["SigQ"])
dataset_25 = pd.DataFrame(columns=["SigPnd"])
dataset_26 = pd.DataFrame(columns=["ShdPnd"])
dataset_27 = pd.DataFrame(columns=["SigBlk"])
dataset_28 = pd.DataFrame(columns=["SigIgn"])
dataset_29 = pd.DataFrame(columns=["SigCgt"])
dataset_30 = pd.DataFrame(columns=["CapInh"])
dataset_31 = pd.DataFrame(columns=["CapPrm"])
dataset_32 = pd.DataFrame(columns=["CapEff"])
dataset_33 = pd.DataFrame(columns=["CapBnd"])
dataset_34 = pd.DataFrame(columns=["Seccomp"])
dataset_35 = pd.DataFrame(columns=["Cpus_allowed"])
dataset_36 = pd.DataFrame(columns=["Cpus_allowed_list"])
dataset_37 = pd.DataFrame(columns=["voluntary_ctxt_switches"])
dataset_38 = pd.DataFrame(columns=["nonvoluntary_ctxt_switches"])

def getParam(line_n, parameter):
    if parameter == "java.lang.":
        if line_n[0:10] == "java.lang.":
            words = line_n.replace('.', '').split()
            error = words[0]
            return error[8:len(error) - 1]
        if line_n[0:len("android.view.WindowManager$")] == "android.view.WindowManager$":
            words = line_n.replace('.', '').split()
            error = words[0]
            return error[len("android.view.WindowManager$")-2:len(error) - 1]
        if line_n[0:len("android.os.")] == "android.os.":
            words = line_n.replace('.', '').split()
            error = words[0]
            return error[len("android.os.") - 2:len(error)]
        if line_n[0:len("android.view.ViewRootImpl$")] == "android.view.ViewRootImpl$":
            words = line_n.replace('.', '').split()
            error = words[0]
            return error[len("android.view.ViewRootImpl$") - 2:len(error) - 1]

    # есть баг Cpus_allowed и Cpus_allowed_list

    elif line_n[0:len(parameter)] == parameter:
        words = line_n.split()
        if parameter == "Uid" or parameter == "Gid":
            return words[1]+" "+words[2]+" "+words[3]+" "+words[4]
        elif parameter == "Groups":
            return words[1]+" "+words[2]+" "+words[3]+" "+words[4]+\
                   " "+words[5]+" "+words[6]+" "+words[7]
        else:
            return words[1]

    else:
        return None

def set_1_ErrorName(line_n, data, n):
    error_name = getParam(line_n, "java.lang.")
    if error_name is not None:
        print(n, error_name)
        data = data.append({'error_name': error_name}, ignore_index=True)
        return data
    return data

def set_2_State(line_n, data, n):
    State = getParam(line_n, "State")
    if State is not None:
        print(n, State)
        data = data.append({'State': State}, ignore_index=True)
        return data
    return data

def set_3_Tgid(line_n, data, n):
    Tgid = getParam(line_n, "Tgid")
    if Tgid is not None:
        print(n, Tgid)
        data = data.append({'Tgid': Tgid}, ignore_index=True)
        return data
    return data

def set_4_Pid(line_n, data, n):
    Pid = getParam(line_n, "Pid")
    if Pid is not None:
        print(n, Pid)
        data = data.append({'Pid': Pid}, ignore_index=True)
        return data
    return data

def set_5_PPid(line_n, data, n):
    PPid = getParam(line_n, "PPid")
    if PPid is not None:
        print(n, PPid)
        data = data.append({'PPid': PPid}, ignore_index=True)
        return data
    return data

def set_6_TracerPid(line_n, data, n):
    TracerPid = getParam(line_n, "TracerPid")
    if TracerPid is not None:
        print(n, TracerPid)
        data = data.append({'TracerPid': TracerPid}, ignore_index=True)
        return data
    return data

def set_7_Uid(line_n, data, n):
    Uid = getParam(line_n, "Uid")
    if Uid is not None:
        print(n, Uid)
        data = data.append({'Uid': Uid}, ignore_index=True)
        return data
    return data

def set_8_Gid(line_n, data, n):
    Gid = getParam(line_n, "Gid")
    if Gid is not None:
        print(n, Gid)
        data = data.append({'Gid': Gid}, ignore_index=True)
        return data
    return data

def set_9_FDSize(line_n, data, n):
    FDSize = getParam(line_n, "FDSize")
    if FDSize is not None:
        print(n, FDSize)
        data = data.append({'FDSize': FDSize}, ignore_index=True)
        return data
    return data

def set_10_Groups(line_n, data, n):
    Groups = getParam(line_n, "Groups")
    if Groups is not None:
        print(n, Groups)
        data = data.append({'Groups': Groups}, ignore_index=True)
        return data
    return data

def set_11_VmPeak(line_n, data, n):
    VmPeak = getParam(line_n, "VmPeak")
    if VmPeak is not None:
        print(n, VmPeak)
        data = data.append({'VmPeak': VmPeak}, ignore_index=True)
        return data
    return data

def set_12_VmSize(line_n, data, n):
    VmSize = getParam(line_n, "VmSize")
    if VmSize is not None:
        print(n, VmSize)
        data = data.append({'VmSize': VmSize}, ignore_index=True)
        return data
    return data

def set_13_VmLck(line_n, data, n):
    VmLck = getParam(line_n, "VmLck")
    if VmLck is not None:
        print(n, VmLck)
        data = data.append({'VmLck': VmLck}, ignore_index=True)
        return data
    return data

def set_14_VmPin(line_n, data, n):
    VmPin = getParam(line_n, "VmPin")
    if VmPin is not None:
        print(n, VmPin)
        data = data.append({'VmPin': VmPin}, ignore_index=True)
        return data
    return data

def set_15_VmHWM(line_n, data, n):
    VmHWM = getParam(line_n, "VmHWM")
    if VmHWM is not None:
        print(n, VmHWM)
        data = data.append({'VmHWM': VmHWM}, ignore_index=True)
        return data
    return data

def set_16_VmRSS(line_n, data, n):
    VmRSS = getParam(line_n, "VmRSS")
    if VmRSS is not None:
        print(n, VmRSS)
        data = data.append({'VmRSS': VmRSS}, ignore_index=True)
        return data
    return data

def set_17_VmData(line_n, data, n):
    VmData = getParam(line_n, "VmData")
    if VmData is not None:
        print(n, VmData)
        data = data.append({'VmData': VmData}, ignore_index=True)
        return data
    return data

def set_18_VmStk(line_n, data, n):
    VmStk = getParam(line_n, "VmStk")
    if VmStk is not None:
        print(n, VmStk)
        data = data.append({'VmStk': VmStk}, ignore_index=True)
        return data
    return data

def set_19_VmExe(line_n, data, n):
    VmExe = getParam(line_n, "VmExe")
    if VmExe is not None:
        print(n, VmExe)
        data = data.append({'VmExe': VmExe}, ignore_index=True)
        return data
    return data

def set_20_VmLib(line_n, data, n):
    VmLib = getParam(line_n, "VmLib")
    if VmLib is not None:
        print(n, VmLib)
        data = data.append({'VmLib': VmLib}, ignore_index=True)
        return data
    return data

def set_21_VmPTE(line_n, data, n):
    VmPTE = getParam(line_n, "VmPTE")
    if VmPTE is not None:
        print(n, VmPTE)
        data = data.append({'VmPTE': VmPTE}, ignore_index=True)
        return data
    return data

def set_22_VmSwap(line_n, data, n):
    VmSwap = getParam(line_n, "VmSwap")
    if VmSwap is not None:
        print(n, VmSwap)
        data = data.append({'VmSwap': VmSwap}, ignore_index=True)
        return data
    return data

def set_23_Threads(line_n, data, n):
    Threads = getParam(line_n, "Threads")
    if Threads is not None:
        print(n, Threads)
        data = data.append({'Threads': Threads}, ignore_index=True)
        return data
    return data

def set_24_SigQ(line_n, data, n):
    SigQ = getParam(line_n, "SigQ")
    if SigQ is not None:
        print(n, SigQ)
        data = data.append({'SigQ': SigQ}, ignore_index=True)
        return data
    return data

def set_25_SigPnd(line_n, data, n):
    SigPnd = getParam(line_n, "SigPnd")
    if SigPnd is not None:
        print(n, SigPnd)
        data = data.append({'SigPnd': SigPnd}, ignore_index=True)
        return data
    return data

def set_26_ShdPnd(line_n, data, n):
    ShdPnd = getParam(line_n, "ShdPnd")
    if ShdPnd is not None:
        print(n, ShdPnd)
        data = data.append({'ShdPnd': ShdPnd}, ignore_index=True)
        return data
    return data

def set_27_SigBlk(line_n, data, n):
    SigBlk = getParam(line_n, "SigBlk")
    if SigBlk is not None:
        print(n, SigBlk)
        data = data.append({'SigBlk': SigBlk}, ignore_index=True)
        return data
    return data

def set_28_SigIgn(line_n, data, n):
    SigIgn = getParam(line_n, "SigIgn")
    if SigIgn is not None:
        print(n, SigIgn)
        data = data.append({'SigIgn': SigIgn}, ignore_index=True)
        return data
    return data

def set_29_SigCgt(line_n, data, n):
    SigCgt = getParam(line_n, "SigCgt")
    if SigCgt is not None:
        print(n, SigCgt)
        data = data.append({'SigCgt': SigCgt}, ignore_index=True)
        return data
    return data

def set_30_CapInh(line_n, data, n):
    CapInh = getParam(line_n, "CapInh")
    if CapInh is not None:
        print(n, CapInh)
        data = data.append({'CapInh': CapInh}, ignore_index=True)
        return data
    return data

def set_31_CapPrm(line_n, data, n):
    CapPrm = getParam(line_n, "CapPrm")
    if CapPrm is not None:
        print(n, CapPrm)
        data = data.append({'CapPrm': CapPrm}, ignore_index=True)
        return data
    return data

def set_32_CapEff(line_n, data, n):
    CapEff = getParam(line_n, "CapEff")
    if CapEff is not None:
        print(n, CapEff)
        data = data.append({'CapEff': CapEff}, ignore_index=True)
        return data
    return data

def set_33_CapBnd(line_n, data, n):
    CapBnd = getParam(line_n, "CapBnd")
    if CapBnd is not None:
        print(n, CapBnd)
        data = data.append({'CapBnd': CapBnd}, ignore_index=True)
        return data
    return data

def set_34_Seccomp(line_n, data, n):
    Seccomp = getParam(line_n, "Seccomp")
    if Seccomp is not None:
        print(n, Seccomp)
        data = data.append({'Seccomp': Seccomp}, ignore_index=True)
        return data
    return data

def set_35_Cpus_allowed(line_n, data, n):
    Cpus_allowed = getParam(line_n, "Cpus_allowed")
    if Cpus_allowed is not None:
        print(n, Cpus_allowed)
        data = data.append({'Cpus_allowed': Cpus_allowed}, ignore_index=True)
        return data
    return data

def set_36_Cpus_allowed_list(line_n, data, n):
    Cpus_allowed_list = getParam(line_n, "Cpus_allowed_list")
    if Cpus_allowed_list is not None:
        print(n, Cpus_allowed_list)
        data = data.append({'Cpus_allowed_list': Cpus_allowed_list}, ignore_index=True)
        return data
    return data

def set_37_Voluntary_ctxt_switches(line_n, data, n):
    voluntary_ctxt_switches = getParam(line_n, "voluntary_ctxt_switches")
    if voluntary_ctxt_switches is not None:
        print(n, voluntary_ctxt_switches)
        data = data.append({'voluntary_ctxt_switches': voluntary_ctxt_switches}, ignore_index=True)
        return data
    return data

def set_38_Nonvoluntary_ctxt_switches(line_n, data, n):
    nonvoluntary_ctxt_switches = getParam(line_n, "nonvoluntary_ctxt_switches")
    if nonvoluntary_ctxt_switches is not None:
        print(n, nonvoluntary_ctxt_switches)
        data = data.append({'nonvoluntary_ctxt_switches': nonvoluntary_ctxt_switches}, ignore_index=True)
        return data
    return data

def createDataset(data_1, data_2, data_3, data_4, data_5, data_6,
                  data_7, data_8, data_9, data_10, data_11, data_12,
                  data_13, data_14, data_15, data_16, data_17, data_18,
                  data_19, data_20, data_21, data_22, data_23, data_24,
                  data_25, data_26, data_27, data_28, data_29, data_30,
                  data_31, data_32, data_33, data_34, data_35, data_36,
                  data_37, data_38):

    for i in range(5620):
        n = i + 1
        error_filename = "CRASH/crash_"+str(n)+".txt"
        f = open(error_filename, "r", encoding = 'utf-8')
        for line in f:
            data_1 = set_1_ErrorName(line, data_1, n)
            data_2 = set_2_State(line, data_2, n)
            data_3 = set_3_Tgid(line, data_3, n)
            data_4 = set_4_Pid(line, data_4, n)
            data_5 = set_5_PPid(line, data_5, n)
            data_6 = set_6_TracerPid(line, data_6, n)
            data_7 = set_7_Uid(line, data_7, n)
            data_8 = set_8_Gid(line, data_8, n)
            data_9 = set_9_FDSize(line, data_9, n)
            data_10 = set_10_Groups(line, data_10, n)
            data_11 = set_11_VmPeak(line, data_11, n)
            data_12 = set_12_VmSize(line, data_12, n)
            data_13 = set_13_VmLck(line, data_13, n)
            data_14 = set_14_VmPin(line, data_14, n)
            data_15 = set_15_VmHWM(line, data_15, n)
            data_16 = set_16_VmRSS(line, data_16, n)
            data_17 = set_17_VmData(line, data_17, n)
            data_18 = set_18_VmStk(line, data_18, n)
            data_19 = set_19_VmExe(line, data_19, n)
            data_20 = set_20_VmLib(line, data_20, n)
            data_21 = set_21_VmPTE(line, data_21, n)
            data_22 = set_22_VmSwap(line, data_22, n)
            data_23 = set_23_Threads(line, data_23, n)
            data_24 = set_24_SigQ(line, data_24, n)
            data_25 = set_25_SigPnd(line, data_25, n)
            data_26 = set_26_ShdPnd(line, data_26, n)
            data_27 = set_27_SigBlk(line, data_27, n)
            data_28 = set_28_SigIgn(line, data_28, n)
            data_29 = set_29_SigCgt(line, data_29, n)
            data_30 = set_30_CapInh(line, data_30, n)
            data_31 = set_31_CapPrm(line, data_31, n)
            data_32 = set_32_CapEff(line, data_32, n)
            data_33 = set_33_CapBnd(line, data_33, n)
            data_34 = set_34_Seccomp(line, data_34, n)
            data_35 = set_35_Cpus_allowed(line, data_35, n)
            data_36 = set_36_Cpus_allowed_list(line, data_36, n)
            data_37 = set_37_Voluntary_ctxt_switches(line, data_37, n)
            data_38 = set_38_Nonvoluntary_ctxt_switches(line, data_38, n)

        f.close()

    dataframe = pd.concat([data_1, data_2, data_3, data_4, data_5, data_6,
                              data_7, data_8, data_9, data_10, data_11, data_12,
                              data_13, data_14, data_15, data_16, data_17, data_18,
                              data_19, data_20, data_21, data_22, data_23, data_24,
                              data_25, data_26, data_27, data_28, data_29, data_30,
                              data_31, data_32, data_33, data_34, data_35, data_36,
                                data_37, data_38], axis=1)

    dataframe.to_csv("./crash_data.csv", index=True)

createDataset(dataset_1, dataset_2, dataset_3, dataset_4, dataset_5, dataset_6,
                              dataset_7, dataset_8, dataset_9, dataset_10, dataset_11, dataset_12,
                              dataset_13, dataset_14, dataset_15, dataset_16, dataset_17, dataset_18,
                              dataset_19, dataset_20, dataset_21, dataset_22, dataset_23, dataset_24,
                              dataset_25, dataset_26, dataset_27, dataset_28, dataset_29, dataset_30,
                              dataset_31, dataset_32, dataset_33, dataset_34, dataset_35, dataset_36,
                                dataset_37, dataset_38)

# def checkFiles():
#     for i in range(5620):
#         n = i+1
#         error_filename = "CRASH/crash_" + str(n) + ".txt"
#         f = open(error_filename, "r", encoding='utf-8')
#         checker = False
#         for line in f:
#             if line[0:len("java.lang.")] == "java.lang.":
#                 checker = True
#             if line[0:len("android.view.WindowManager$")] == "android.view.WindowManager$":
#                 checker = True
#         if not checker:
#             print(n)
#             copyfile(error_filename,"crash_valid_but_1/crash_" + str(n) + ".txt")
#
#         f.close()
#
# checkFiles()

# for i in range(5620):
#     n = i + 1
#     error_filename = "CRASH/crash_" + str(n) + ".txt"
#     f = open(error_filename, "r", encoding='utf-8')
#     checker = False
#     name = ""
#     for line in f:
#         #dataset = setParam(line, dataset, n)
#         if line[0:10] == "java.lang.":
#             words = line.replace('.', '').split()
#             error = words[0]
#             name += error[8:len(error) - 1]
#         if line[0:len("android.view.WindowManager$")] == "android.view.WindowManager$":
#             words = line.replace('.', '').split()
#             error = words[0]
#             name += error[len("android.view.WindowManager$")-2:len(error) - 1]
#
#         if len(name) >= len("ArrayIndexOutOfBounds"):
#             checker = True
#     if checker:
#         print(n)
#
#     f.close()