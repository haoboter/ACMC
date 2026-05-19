/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// File      : 626-826-migrate.c
// Function  : Migration aid for porting Sensoray 626 apps to model 826
// Copyright : (C) 2022 Sensoray
//
// This file is intended as an aid for porting 626 apps to model 826.
// It provides a set of 826-compatible equivalents for functions in the 626 API.
//
// This is not a plug-and-play replacement for the 626 API -- it's merely a tool for speeding up the porting process.
// When you run your app, replacement functions will report warnings to help you find code that must be changed.
// The warnings and source comments offer porting hints, but developers are strongly advised to consult the API
// documentation to ensure complete understanding and facilitate effective resolution of specific porting issues.
//
// Notes:
//  * USE AT YOUR OWN RISK! This code has not been extensively tested, nor is it guaranteed to be error-free or to
//     cover all possible app scenarios. If an error is discovered, please report it to Sensoray.
//  * In functions that have argument "hbd" (board handle), the hbd value (0 to 15) must match the address switch
//     settings on the target 826 board.
//  * The arguments and return value used in a replacement function may differ those of its 626 API equivalent.
//     These differences will cause compiler errors when the app is built, thus serving notice that there are
//     porting issues that need attention.
//  * Users should implement additional error checking/handling as appropriate. In particular, many of the
//     replacement functions return an 826 error code that should be checked by the calling function.
//
// To port an application from 626 to 826:
//  * Install the 826 API and driver (626 API and driver are not used and may be removed if desired).
//  * Remove include files from app sources: app626.h, win626.h
//  * Add include files to app sources: 826api.h, port626.h
//  * Remove source from app project: win626.c
//  * Add source to app project: 626-826-migrate.c
//  * Add linker dependency to app project: s826.lib
//  * Edit app sources as necessary to conform to replacement function prototypes.
//  * Build the app and resolve compiler errors.
//  * Run the app and heed warnings issued by the replacement functions.
//  * Modify the app source and this file as needed to resolve errors and make the app function properly.
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program. If not, see <https://www.gnu.org/licenses/>.
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>      // printf for warnings related to porting issues

#include "826api.h"     // 826 API standard header
#include "port626.h"    // definitions needed for this code migration module

////////////////////////////////////////////////////////////////////
// Issue notification about a porting issue.
// Comment out one of the following lines.

#define WARN(MSG)   printf("warning: %s\n", MSG)    // issue runtime warning
//#define WARN(MSG)   error msg                     // throw a compiler error

//////////////////////////////////////////////////////////////////////////////
// Execute 826 API function and, if error detected, return errcode.

#define api(apifunc)            \
    errcode = (apifunc);        \
    if (errcode != S826_ERR_OK) \
        return errcode;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////  ADMIN  //////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct VER {    // version info returned by S826_VersionRead()
    uint api;   // API
    uint drv;   // device driver
    uint bd;    // board rev
    uint ga;    // FPGA
} VER;

static int detflags = 0;    // List of 826 boards detected by S826_SystemOpen(), one flag per board: '1'=detected, '0'=undetected. Bit index matches board address switches.

ERR826 GetVersions(HBD hbd, VER *ver)
{
    return S826_VersionRead(hbd, &ver->api, &ver->drv, &ver->bd, &ver->ga);
}

uint S626_GetDllVersion(HBD hbd)
{
    VER ver;
    return (GetVersions(hbd, &ver) == S826_ERR_OK) ? ver.api : 0;
}

uint S626_GetDriverVersion(HBD hbd)
{
    VER ver;
    return (GetVersions(hbd, &ver) == S826_ERR_OK) ? ver.drv : 0;
}

void S626_GetErrors(HBD hbd)
{
    WARN("S626_GetErrors() has no 826 equivalent. Note: 826 API functions return an error code");
}

ERR826 S626_DLLOpen(void)
{
    int rval = S826_SystemOpen();   // open 826 API.
    if (rval < 0) {                 // if error
        detflags = 0;               //   indicate no boards detected
        return rval;                //   return 826 error code.
    }                               // else
    detflags = rval;                //   save flags that indicate detected 826 boards, one flag per valid board address in range [0:15] (as set by board's address switches).
    return S826_ERR_OK;             //   indicate API opened without errors.
}

void S626_DLLClose(void)
{
    detflags = 0;           // mark all 826 boards closed
    S826_SystemClose();     // close the API.
}

ERR826 S626_OpenBoard(HBD hbd, uint address, void *callback, u16 priority)    // Note: address and priority are ignored
{
    // 626: opens one 626 board and defines a callback function for interrupt processing.
    // 826: S826_SystemOpen() detects and opens all 826 boards. Note: blocking functions are used in lieu of callbacks for interrupt processing.

    if (callback != NULL) {
        WARN("826 API uses blocking functions for interrupt processing (callbacks are deprecated)");
        return S826_ERR_VALUE;
    }
    else if ((detflags & (1 << hbd)) == 0)
        return S826_ERR_NOTREADY;   // 826 board wasn't detected
    else
        return S826_ERR_OK;  // board was opened by S826_SystemOpen(), so nothing else to do here
}

void S626_CloseBoard(HBD hbd)
{
    // board will be closed via S826_SystemClose() -- nothing needs to be done here
}

uint S626_GetAddress(HBD hbd)
{
    return hbd;   // irrelevant on 826; 826 address switch must match hbd
}

// Direct 826 register read/write is not supported.
u16 S626_RegRead(HBD hbd, u16 regadrs)
{
    WARN("S626_RegRead() has no 826 equivalent");
    return 0;
}

void S626_RegWrite(HBD hbd, u16 regadrs, u16 value)
{
    WARN("S626_RegWrite() has no 826 equivalent");

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////  INTERRUPTS  ////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// 626 employs a single user-defined callback function for interrupt handling.
// Upon any 626 IRQ, the 626 device driver will invoke the callback function in a private, independent context.
// The callback function must determine the interrupt source and handle the interrupt accordingly.
// The callback function is responsible for marshalling interrupt events into their associated app thread contexts.

// 826 employs 4 built-in blocking API functions for interrupt handling: S826_CounterSnapshotRead(), S826_AdcRead(), S826_DioCapRead() and S826_WatchdogEventWait().
// Upon any 826 IRQ, the 826 device driver will determine the interrupt source and unblock the associated blocking API function. An app thread waits for a
// specific interrupt by calling the appropriate blocking function; consequently, the interrupt event is inherently marshalled into the associated app thread context.

void S626_InterruptEnable(HBD hbd, u16 enable)
{
    // 826: the API will automatically enable a specific interrupt when the associated blocking function is entered, and disable it when the function exits.

    WARN("S626_InterruptEnable() has no 826 equivalent; see blocking functions in 826 API");
}

void S626_InterruptStatus(HBD hbd, u16 *status)
{
    // On 826, IRQ status can be polled by calling the associated blocking function with tmax = 0.
    // The blocking function's return value indicates interrupt status:
    //  * S826_ERR_OK       - interrupt occurred.
    //  * S826_ERR_NOTREADY - no interrupt.
    // Some blocking functions can be unblocked by more than one IRQ. To determine the IRQ that caused the unblock:
    //  * S826_CounterSnapshotRead() - captured events are listed in the returned snapshot's "reason" flags.
    //  * S826_DioCapRead()          - captured events are listed in the returned chanlist[] flags.

    WARN("S626_InterruptStatus() has no 826 equivalent; see blocking functions in 826 API");
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////  ADC  ////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static uint AdcSlotlist[16];    // list of active adc slots for up to sixteen 826 boards; for each board, one bit per slot (1=active)

ERR826 S626_ResetADC(HBD hbd, u8 *pollist)  // poll list item: (EOPL, 0, 0, RNG5/RNG10_N, CHAN<3:0>)
{
    uint tsettle = 15;              // per-channel settling time in microseconds; change this as desired

    ERR826 errcode;
    u8 item;
    uint chan, range;
    uint slotnum = 0;               // assign slots sequentially starting at slot0
    AdcSlotlist[hbd] = 0;           // start with empty slot list

    api(S826_AdcEnableWrite(hbd, 0));    // disable adc

    do {                                                                    // Translate and install poll list:
        item = pollist[slotnum];                                            //   cache poll list entry
        chan = item & ADC_CHANMASK;                                         //   extract AIN channel number
        range = (item & ADC_RANGE_5V) ? S826_ADC_GAIN_2 : S826_ADC_GAIN_1;  //   translate range code
        api(S826_AdcSlotConfigWrite(hbd, slotnum, chan, tsettle, range));   //   install translated poll list entry
        AdcSlotlist[hbd] |= (1 << slotnum++);                               //   mark slot active and bump slot count
    } while ((item & ADC_EOPL) == 0);                                       //   loop while not end of poll list
        
    api(S826_AdcSlotlistWrite(hbd, AdcSlotlist[hbd], 0));   // translate EOPL
    api(S826_AdcTrigModeWrite(hbd, S826_INSRC_VDIO(0)));    // use VirtualDio0 rising edge as adc trigger
    api(S826_AdcEnableWrite(hbd, 1));                       // enable adc
    return S826_ERR_OK;
}

#define VDIO_0_MASK     (1 << 0)

ERR826 S626_StartADC(HBD hbd)
{
    ERR826 errcode;
    api(S826_VirtualWrite(hbd, VDIO_0_MASK, S826_BITSET));        // trigger adc conversion burst (assert VirtualDio0)
    api(S826_VirtualWrite(hbd, VDIO_0_MASK, S826_BITCLR));        // reset trigger (negate VirtualDio0)
    return S826_ERR_OK;
}

ERR826 S626_WaitDoneADC(HBD hbd, s16 databuf[16])
{
    ERR826 errcode;
    int i, buf[16];
    uint slotlist = AdcSlotlist[hbd];
    api(S826_AdcRead(hbd, buf, NULL, &slotlist, S826_WAIT_INFINITE));   // block until adc burst completes
    for (i = 0; i < 15; i++)
        databuf[i] = (s16)(buf[i] & 0xFFFF);                            // copy adc samples to databuf
    return S826_ERR_OK;
}

ERR826 S626_ReadADC(HBD hbd, s16 databuf[16])
{
    ERR826 errcode;
    api(S626_StartADC(hbd));                // start adc conversion burst
    api(S626_WaitDoneADC(hbd, databuf));    // wait for conversions to complete and copy adc samples to databuf[]
    return S826_ERR_OK;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////  DAC  ////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ERR826 S626_WriteDAC(HBD hbd, u16 chan, int setpoint)  // setpoint: 0 = 0V; -8191 = -10V; +8191 = +10V
{
    ERR826 errcode;
    if ((chan > 3) || (setpoint < -8191) || (setpoint > 8191))
        return S826_ERR_VALUE;
    else {
        uint sp = (uint)((setpoint + 8191) * 65535.0 / 16382.0);    // translate dac data
        api(S826_DacRangeWrite(hbd, chan, 3, 0));                   // select +/-10V output range
        api(S826_DacDataWrite(hbd, chan, sp, 0));                   // program analog output
        return S826_ERR_OK;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////  WATCHDOG  /////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define WD826_TOGGLE_SCALAR     25000
#define WD826_INTERVAL_SCALAR   50000

#define WD626_PERIOD_125MS      0
#define WD626_PERIOD_500MS      1
#define WD626_PERIOD_1S         2
#define WD626_PERIOD_10S        3

ERR826 S626_WatchdogPeriodSet(HBD hbd, u16 periodEnum)
{
    uint ms, timers[5];
    switch (periodEnum) {                                                       // translate 626 enum value to milliseconds
        case WD626_PERIOD_125MS: ms = 125;   break;
        case WD626_PERIOD_500MS: ms = 500;   break;
        case WD626_PERIOD_1S:    ms = 1000;  break;
        case WD626_PERIOD_10S:   ms = 10000; break;
        default: return S826_ERR_VALUE;
    }
    timers[S826_WD_DELAY0] = ms * WD826_INTERVAL_SCALAR;                        // populate timing info struct
    timers[S826_WD_DELAY1] = timers[S826_WD_DELAY2] = 1;
    timers[S826_WD_PWIDTH] = timers[S826_WD_PGAP] = ms * WD826_TOGGLE_SCALAR;
    return S826_WatchdogConfigWrite(hbd, S826_WD_PEN, timers);                  // config watchdog; set PEN for 626 compatibility
}

ERR826 S626_WatchdogPeriodGet(HBD hbd, u16 *periodEnum)
{
    uint cfg, timers[5];
    ERR826 errcode;
    api(S826_WatchdogConfigRead(hbd, &cfg, timers));
    switch (timers[0] / WD826_INTERVAL_SCALAR) {
        case 125:  *periodEnum = WD626_PERIOD_125MS; break;
        case 500:  *periodEnum = WD626_PERIOD_500MS; break;
        case 1000: *periodEnum = WD626_PERIOD_1S;    break;
        default: return S826_ERR_VALUE;  // doesn't match any 626 enum value
    }
    return S826_ERR_OK;
}

ERR826 S626_WatchdogEnableSet(HBD hbd, uint enable)
{
    return S826_WatchdogEnableWrite(hbd, enable);
}

ERR826 S626_WatchdogEnableGet(HBD hbd, uint *enable)
{
    return S826_WatchdogEnableRead(hbd, enable);
}

ERR826 S626_WatchdogReset(HBD hbd)
{
    return S826_WatchdogKick(hbd, 0X5A55AA5A);
}

ERR826 S626_WatchdogTimeout(HBD hbd, u16 *timeout)
{
    ERR826 errcode = S826_WatchdogEventWait(hbd, 0);
    switch (errcode) {
        case S826_ERR_OK:       *timeout = 1; break;    // timed out
        case S826_ERR_NOTREADY: *timeout = 0; break;    // not timed out
        default:                return errcode;         // error (e.g., wait was cancelled by another thread)
    }
    return S826_ERR_OK;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////  BATTERY  ////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

u16 S626_BackupEnableGet(HBD hbd)
{
    WARN("S626_BackupEnableGet() has no 826 equivalent");
    return 0;
}

void S626_BackupEnableSet(HBD hbd, u16 en)
{
    WARN("S626_BackupEnableSet() has no 826 equivalent");
}

u16 S626_ChargeEnableGet(HBD hbd)
{
    WARN("S626_ChargeEnableGet() has no 826 equivalent");
    return 0;
}

void S626_ChargeEnableSet(HBD hbd, u16 en)
{
    WARN("S626_ChargeEnableSet() has no 826 equivalent");
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////  GPIO  ///////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// 626 operates on one 16-bit GPIO group at a time; 826 operate on a uint array in which each uint is associated with 24 GPIOs.
// GPIO grouping:
// 626: (47:40, 39:32)(31:24, 23:16)(15:8, 7:0) = (group2)(group1)(group0)
// 826: (47:40, 39:32, 31:24)(23:16, 15:8, 7:0) = (buf[1])(buf[0])

static ERR826 ExtractDioGroup(HBD hbd, u16 group, uint buf[2], u16 *data)
{
    switch (group) {    // Extract 626-compatible 16-bit group from 826-compatible buf[2].
        case 0: 
            *data = (u16)buf[0]; 
            break;
        case 1: 
            *data = (u16)((buf[1] & 0x000000FF) << 8) | ((buf[0] & 0x00FF0000) >> 16); 
            break;
        case 2: 
            *data = (u16)((buf[1] & 0x00FFFF00) >> 8); 
            break;
        default:
            return S826_ERR_VALUE;
    }
    return S826_ERR_OK;
}

static ERR826 InsertDioGroup(HBD hbd, u16 group, uint buf[2], u16 data)
{
    switch (group) {    // Insert 626-compatible 16-bit group into 826-compatible buf[2].
        case 0:
            buf[0] = (buf[0] & 0x00FF0000) | data;
            break;
        case 1:
            buf[0] = (buf[0] & 0x0000FFFF) | (uint)(data & 0x00FF) << 16;
            buf[1] = (buf[1] & 0x00FFFF00) | (uint)(data >> 8);
            break;
        case 2:
            buf[1] = (buf[1] & 0x000000FF) | (uint)data << 8;
            break;
        default:
            return S826_ERR_VALUE;
    }
    return S826_ERR_OK;
}

ERR826 S626_DIOWriteBankGet(HBD hbd, u16 group, u16 *data)
{
    uint buf[2];
    ERR826 errcode;
    api(S826_DioOutputRead(hbd, buf));                  // read 48 output states
    api(ExtractDioGroup(hbd, group, buf, data));        // extract 16-bit group
    return S826_ERR_OK;
}

ERR826 S626_DIOReadBank(HBD hbd, u16 group, u16 *data)
{
    uint buf[2];
    ERR826 errcode;
    api(S826_DioInputRead(hbd, buf));                   // read 48 input states
    api(ExtractDioGroup(hbd, group, buf, data));        // extract 16-bit group
    return S826_ERR_OK;
}

ERR826 S626_DIOWriteBankSet(HBD hbd, u16 group, u16 data)
{
    ERR826 errcode;

    uint set[2] = {0, 0};
    uint clr[2] = {0, 0};
    api(InsertDioGroup(hbd, group, set, data));         // create 48-bit mask of bits to set
    api(InsertDioGroup(hbd, group, clr, ~data));        // create 48-bit mask of bits to clear
    api(S826_DioOutputWrite(hbd, set, S826_BITSET));    // set all outputs specified as '1' in data
    api(S826_DioOutputWrite(hbd, clr, S826_BITCLR));    // clear all outputs specified as '0' in data

    // Note: The above bit set/clear operations provide thread-safe operation without a critical section. However, they may not modify all GPIOs
    // in the group at the same time (because 0->1 transitions occur before 1->0 transitions). If the app requires simultaneous set/clear of all
    // GPIOs in the group, replace the above with an atomic read-modify-write sequence.
    // Example:

    //  uint buf[2];
    //  CriticalSectionBegin();                             // start atomic op
    //  api(S826_DioOutputRead(hbd, buf));                  // read 48 output states into buf
    //  api(InsertDioGroup(hbd, group, buf, data));         // modify -- replace buf group with new data
    //  api(S826_DioOutputWrite(hbd, clr, S826_BITWRITE));  // write buf to 48 GPIOs
    //  CriticalSectionEnd();                               // end atomic op

    return S826_ERR_OK;
}

void S626_DIOModeSet(HBD hbd, u16 chan, u16 value)
{
    WARN("S626_DIOModeSet() has no 826 equivalent; see S826_DioOutputSourceWrite()");
}

u16 S626_DIOModeGet(HBD hbd, u16 group)
{
    WARN("S626_DIOModeGet() has no 826 equivalent; see S826_DioOutputSourceWrite()");
    return 0;
}

void S626_DIOEdgeSet(HBD hbd, u16 group, u16 value)
{
    // 626: S626_DIOEdgeSet() sets edge polarity (rising or falling) for all GPIOs in a group. Note: capture enables are not changed.
    // 826: S826_DioCapEnablesRead() sets capture enables (rising, falling, both or neither) for all 48 GPIOs. Note: edge polarities are inherently specified by capture enables.
    WARN("S626_DIOEdgeSet() has no 826 equivalent; see S826_DioCapEnablesWrite()");
}

u16 S626_DIOEdgeGet(HBD hbd, u16 group)
{
    // 626: S626_DIOEdgeGet() returns edge polarity (rising or falling) for all GPIOs in a group. Note: this does not indicate whether capturing is enabled.
    // 826: S826_DioCapEnablesRead() returns capture enables (rising, falling, both or neither) for all 48 GPIOs.
    WARN("S626_DIOEdgeGet() has no 826 equivalent; see S826_DioCapEnablesRead()");
    return 0;
}

void S626_DIOCapEnableSet(HBD hbd, u16 group, u16 chanmask, u16 enable)
{
    // 626: S626_DIOCapEnableSet() enables or disables edge capturing for all selected GPIOs in a group. Note: edge polarity is specified by S626_DIOEdgeSet().
    // 826: S826_DioCapEnablesWrite() sets individual edge capture enables (rising, falling, both or neither) for all 48 GPIOs.
    WARN("S626_DIOCapEnableSet() has no 826 equivalent; see S826_DioCapEnablesWrite()");
}

u16  S626_DIOCapEnableGet(HBD hbd, u16 group)
{
    // 626: S626_DIOCapEnableGet() returns capture enables for all GPIOs in a group.
    // 826: S826_DioCapEnablesRead() returns individual capture enables (rising, falling, both or neither) for all 48 GPIOs.
    WARN("S626_DIOCapEnableGet() has no 826 equivalent; see S826_DioCapEnablesRead()");
    return 0;
}

u16 S626_DIOCapStatus(HBD hbd, u16 group)
{
    // 626: S626_DIOCapStatus() returns capture status for all GPIOs in a group.
    // 826: S826_DioCapRead() blocks until capture or timeout; returns capture status for all 48 GPIOs.
    WARN("S626_DIOCapStatus() has no 826 equivalent; see S826_DioCapRead()");
    return 0;
}

void S626_DIOCapReset(HBD hbd, u16 group, u16 value)
{
    // 626: S626_DIOCapReset() resets individual capture flags
    // 826: S826_DioCapRead() automatically resets capture flags.
    WARN("S626_DIOCapReset() has no 826 equivalent; see S826_DioCapRead()");
}

u16 S626_DIOIntEnableGet(HBD hbd, u16 group)
{
    WARN("S626_DIOIntEnableGet() has no 826 equivalent; see S826_DioCapRead()");
    return 0;
}

void S626_DIOIntEnableSet(HBD hbd, u16 group, u16 value)
{
    WARN("S626_DIOIntEnableSet() has no 826 equivalent; see S826_DioCapRead()");
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////  COUNTERS  //////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ERR826 S626_CounterSoftIndex(HBD hbd, u16 chan)
{
    return S826_CounterPreload(hbd, chan, 0, 0);    // software trigger: captures a snapshot (count, timestamp and reason flags) to the counter's snapshot fifo
}

u16 S626_CounterCapStatus(HBD hbd)
{
    // 626: this function returns event capture flags, which indicate why the counter was sampled to the counter's latch register.
    // 826: S826_CounterSnapshotRead() blocks until snapshot captured or timeout; a snapshot contains "reason flags" which indicate why it was sampled to the counter's event fifo.
    WARN("S626_CounterCapStatus() has no 826 equivalent; see S826_CounterSnapshotRead()");
    return 0;
}

void S626_CounterCapFlagsReset(HBD hbd, u16 chan)
{
    // 626: resets event capture flags in the counter's status register.
    // 826: this function is not needed -- status bits ("reason flags") are captured (in each snapshot) along with count and therefore don't require resetting.
    WARN("S626_CounterCapFlagsReset() has no 826 equivalent; see S826_CounterSnapshotRead() for related functionality");
}

void S626_CounterLoadTrigSet(HBD hbd, u16 chan, u16 value)
{
    // 626: selects the events that will cause counter to be loaded from the preload register.
    // 826: in S826_CounterModeSet(), the TP, NR and BP fields determine preload behavior. Note: 826 has 2 preload registers.
    WARN("S626_CounterLoadTrigSet() has no 826 equivalent; see S826_CounterModeWrite() TP, NR and BP fields");
    // Example:
    S826_CounterModeWrite(hbd, chan, S826_CM_PX_START | S826_CM_PX_ZERO); // example: preload once when counter becomes enabled, and whenever counter overflows/underflows.
}

ERR826 S626_CounterPreload(HBD hbd, u16 chan, uint value)
{
    return S826_CounterPreloadWrite(hbd, chan, 0, value);    // Use 826 Preload0 register. Note: 626 has 24-bit counters; 826 has 32-bit counters.
}

ERR826 S626_CounterReadLatch(HBD hbd, u16 chan, uint *counts)
{
    // 626: this function returns sampled counter value stored in counter's latch register.
    // 826: S826_CounterSnapshotRead() returns oldest snapshot stored in counter's event fifo. It will optionally block if no snapshot is available.
    WARN("S626_CounterReadLatch() has no 826 equivalent; see S826_CounterSnapshotRead()");
    return S826_CounterSnapshotRead(hbd, chan, counts, NULL, NULL, 0);     // using tmax = 0 so function will never block -- to emulate 626 behavior
}

#define S826_TE_MASK    (3 << 9)
#define S826_TD_MASK    (3 << 7)

ERR826 S626_CounterEnableSet(HBD hbd, u16 chan, u16 enable)
{
    ERR826 errcode;
    uint mode;                                      // read-modify-write the counter mode:
    api(S826_CounterModeRead(hbd, chan, &mode));            // read
    mode &= ~(S826_TE_MASK | S826_TD_MASK);                 // modify
    if (enable == CLKENAB_ALWAYS)
        mode |= S826_CM_TE_STARTUP;                             // enable counting when counter becomes enabled
    else { // CLKENAB_INDEX
        mode |= (S826_CM_TE_IXRISE | S826_CM_TD_IXFALL),        // enable counting upon index rising edge, disable upon index falling edge
        WARN("check index polarity");
    }
    return S826_CounterModeWrite(hbd, chan, mode);          // write
}

void S626_CounterIntSourceSet(HBD hbd, u16 chan, u16 intsrcEnum)
{
    // 626: this function enables any combination of these interrupt sources: underflow/overflow, active index edge.
    // 826: this function not needed because a counter has only one interrupt source: Captured snapshot in fifo.
    //  * Interrupts are automatically generated when a snapshot is captured.
    //  * S826_CounterSnapshotRead() automatically enables interrupt upon entry and disables it upon exit. It blocks until snapshot captured or timeout, whichever comes first.
    //  * S826_CounterSnapshotConfigWrite() specifies events that trigger snapshots.
    //  * Snapshots may be triggered by any combination of these: underflow/overflow, index rising/falling edges, ExtIn rising/falling edges, Compare0/Compare1 matches, quadrature error, soft trigger.

    WARN("S626_CounterIntSourceSet() has no 826 equivalent; see S826_CounterSnapshotConfigWrite()");
    // Example:
    S826_CounterSnapshotConfigWrite(hbd, chan, S826_SS_ALL_ZERO | S826_SS_ALL_IXRISE, S826_BITWRITE);  // example: enable snapshot capture upon underflow/overflow or index rising edge
}

ERR826 S626_CounterLatchSourceSet (HBD hbd, u16 chan, u16 intsrcEnum)
{
    // intsrcEnum specifies events that cause the count to be sampled.
    // 626: count is sampled to a latch register upon one of these: A/B upon reading A/B latch; A upon A index; B upon B index; B upon A overflow.
    // 826: count is sampled along with timestamp and reason flags to create a "snapshot"; snapshots are stored in a 16-deep fifo.
    //  * A snapshot is captured upon any combination of these: underflow/overflow, index rising/falling edges, ExtIn rising/falling edges, Compare0/Compare1 register matches, quadrature error, soft trigger.
    //  * Use S826_CounterSnapshotConfigWrite() to specify events that cause a snapshot to be captured:

    switch (intsrcEnum) {           // translate 626 enum
        case LATCHSRC_AB_READ:
            return S826_CounterSnapshotConfigWrite(hbd, chan, 0, S826_BITWRITE);  // no sampling triggers used
        case LATCHSRC_A_INDXA:
            WARN("check polarity of index signal");  // may need to use S826_SS_ALL_IXFALL, depending on index polarity
            return S826_CounterSnapshotConfigWrite(hbd, chan, S826_SS_ALL_IXRISE, S826_BITWRITE);
        case LATCHSRC_B_INDXB:
            // 626: counters come in A/B pairs; the board has 3 A/B pairs. A and B have different architectures and capabilities.
            // 826: all counters are identical; the board has 6 general-purpose counters.
            WARN("LATCHSRC_B_INDXB: consider using S826_SS_ALL_IXRISE on a different counter channel");
            return S826_ERR_VALUE;
        case LATCHSRC_B_OVERA:
            // 626: counters come in A/B pairs; the board has 3 A/B pairs. A and B have different architectures and capabilities.
            // 826: all counters are identical; the board has 6 general-purpose counters.
            WARN("LATCHSRC_B_OVERA: consider using ExtIn signal to connect output of one counter to input of another counter");
            return S826_ERR_VALUE;
    }
    return S826_ERR_VALUE;
}

// Due to architectural differences between 626 and 826 counters, the counter mode words have significant differences which
// make direct translation impractical. In the following list, a related 826 resource is given for each field in the 626 mode word:
//
// 626 ----    See this in 826 API ------
//  LatchSrc    S826_CounterSnapshotConfigWrite() -- defines events which cause snapshots to be captured to counter's event fifo
//  IntSrc      S826_CounterSnapshotConfigWrite() -- defines events which cause snapshots, which in turn generate interrupts
//  LoadSrc     S826_CounterModeWrite(), mode word, TP/NR/BP fields
//  IndxSrc     S826_CounterModeWrite(), mode word, XS field 
//  IndxPol     S826_CounterModeWrite(), mode word, XS field
//  ClkSrc      S826_CounterModeWrite(), mode word, K field
//  ClkPol      S826_CounterModeWrite(), mode word, K field
//  ClkMult     S826_CounterModeWrite(), mode word, K field
//  ClkEnab     S826_CounterModeWrite(), mode word, TE/TD fields

void S626_CounterModeSet(HBD hbd, u16 chan, u16 mode)
{
    WARN("S626_CounterModeSet() has no direct 826 equivalent; see comments for porting tips");
}

u16 S626_CounterModeGet(HBD hbd, u16 chan)
{
    WARN("S626_CounterModeGet() has no direct 826 equivalent; see comments for porting tips");
    return 0;
}
