///////////////////////////////////////////////////////////////////////////////
// Module    : port626.h
// Function  : Header file for porting Sensoray 626 applications to 826.
// Copyright : (C) 2022 Sensoray
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
///////////////////////////////////////////////////////////////////////////////

#ifndef _INC_PORT626_H_
#define _INC_PORT626_H_

#ifdef __cplusplus
extern "C" {
#endif

// ADC poll list constants *************************************************************************

#define ADC_EOPL            0x80                // End-Of-Poll-List marker.
#define ADC_RANGE_10V       0x00                // Range code for ±10V range.
#define ADC_RANGE_5V        0x10                // Range code for ±5V range.
#define ADC_CHANMASK        0x0F                // Channel number mask.

// Counter constants *******************************************************************************

                                                // LoadSrc values:
#define LOADSRC_INDX        0                   //    Preload core in response to Index.
#define LOADSRC_OVER        1                   //    Preload core in response to Overflow.
#define LOADSRCB_OVERA      2                   //    Preload B core in response to A Overflow.
#define LOADSRC_NONE        3                   //    Never preload core.

                                                // IntSrc values:
#define INTSRC_NONE         0                   //    Interrupts disabled.
#define INTSRC_OVER         1                   //    Interrupt on Overflow.
#define INTSRC_INDX         2                   //    Interrupt on Index.
#define INTSRC_BOTH         3                   //    Interrupt on Index or Overflow.

                                                // LatchSrc values:
#define LATCHSRC_AB_READ    0                   //    Latch on read.
#define LATCHSRC_A_INDXA    1                   //    Latch A on A Index.
#define LATCHSRC_B_INDXB    2                   //    Latch B on B Index.
#define LATCHSRC_B_OVERA    3                   //    Latch B on A Overflow.

                                                // IndxSrc values:
#define INDXSRC_HARD        0                   //    Hardware or software index.
#define INDXSRC_SOFT        1                   //    Software index only.

                                                // IndxPol values:
#define INDXPOL_POS         0                   //    Index input is active high.
#define INDXPOL_NEG         1                   //    Index input is active low.

                                                // ClkSrc values:
#define CLKSRC_COUNTER      0                   //    Counter mode.
#define CLKSRC_TIMER        2                   //    Timer mode.
#define CLKSRC_EXTENDER     3                   //    Extender mode.

                                                // ClkPol values:
#define CLKPOL_POS          0                   //    Counter/Extender clock is active high.
#define CLKPOL_NEG          1                   //    Counter/Extender clock is active low.
#define CNTDIR_UP           0                   //    Timer counts up.
#define CNTDIR_DOWN         1                   //    Timer counts down.

                                                // ClkEnab values:
#define CLKENAB_ALWAYS      0                   //    Clock always enabled.
#define CLKENAB_INDEX       1                   //    Clock is enabled by index.

                                                // ClkMult values:
#define CLKMULT_4X          0                   //    4x clock multiplier.
#define CLKMULT_2X          1                   //    2x clock multiplier.
#define CLKMULT_1X          2                   //    1x clock multiplier.

                                                // Bit Field positions in COUNTER_SETUP structure:
#define BF_LOADSRC          9                   //    Preload trigger.
#define BF_INDXSRC          7                   //    Index source.
#define BF_INDXPOL          6                   //    Index polarity.
#define BF_CLKSRC           4                   //    Clock source.
#define BF_CLKPOL           3                   //    Clock polarity/count direction.
#define BF_CLKMULT          1                   //    Clock multiplier.
#define BF_CLKENAB          0                   //    Clock enable.

                                                // Counter channel numbers:
#define CNTR_0A             0                   //    Counter 0A.
#define CNTR_1A             1                   //    Counter 1A.
#define CNTR_2A             2                   //    Counter 2A.
#define CNTR_0B             3                   //    Counter 0B.
#define CNTR_1B             4                   //    Counter 1B.
#define CNTR_2B             5                   //    Counter 2B.

// Counter overflow/index event flag masks for S626_CounterCapStatus() and S626_InterruptStatus().
#define INDXMASK(C)        ( 1 << ( ( (C) > 2 ) ? ( (C) * 2 - 1 ) : ( (C) * 2 +  4 ) ) )
#define OVERMASK(C)        ( 1 << ( ( (C) > 2 ) ? ( (C) * 2 + 5 ) : ( (C) * 2 + 10 ) ) )


// Types --------------------------------------------------------------

typedef unsigned char   u8;
typedef unsigned short  u16;
typedef short           s16;
typedef uint            HBD;        // Board handle in range [0:15]. This must match the address switch settings on the target 826 board.
typedef int             ERR826;     // 826 error code.

#ifndef NULL
#define NULL    ((void *)0)
#endif


// Prototypes ------------------------------------------------------------

// These are the prototypes of "replacement" functions in 626-826-migrate.c, which replace the "original" functions implemented in s626.dll.
//
// Error handling:
//  Note that the prototype of a replacement function may not match that of the equivalent original function.
//  In particular, most replacement functions directly return an 826 error code, whereas the original functions typically set
//  an error code and expect the app to call S626_GetErrors() to check for errors. 
//  

// Admin
ERR826  S626_DLLOpen                (void);
void    S626_DLLClose               (void);
uint    S626_GetAddress             (HBD hbd);
void    S626_GetErrors              (HBD hbd);
uint    S626_GetDllVersion          (HBD hbd);
uint    S626_GetDriverVersion       (HBD hbd);
ERR826  S626_OpenBoard              (HBD hbd, uint PhysLoc, void *IntFunc, u16 Priority);
void    S626_CloseBoard             (HBD hbd);
void    S626_InterruptEnable        (HBD hbd, u16 enable);
void    S626_InterruptStatus        (HBD hbd, u16 *status);

// Diagnostics
u16     S626_RegRead                (HBD hbd, u16 addr);
void    S626_RegWrite               (HBD hbd, u16 addr, u16 value);

// Analog I/O
ERR826  S626_ReadADC                (HBD hbd, s16 *databuf);
ERR826  S626_ResetADC               (HBD hbd, u8 *pollist);
ERR826  S626_StartADC               (HBD hbd);
ERR826  S626_WaitDoneADC            (HBD hbd, s16 *databuf);
ERR826  S626_WriteDAC               (HBD hbd, u16 chan, int value);

// GPIO
ERR826  S626_DIOReadBank            (HBD hbd, u16 group, u16 *data);
ERR826  S626_DIOWriteBankGet        (HBD hbd, u16 group, u16 *data);
ERR826  S626_DIOWriteBankSet        (HBD hbd, u16 group, u16 data);

u16     S626_DIOEdgeGet             (HBD hbd, u16 group);
void    S626_DIOEdgeSet             (HBD hbd, u16 group, u16 value);
void    S626_DIOCapEnableSet        (HBD hbd, u16 group, u16 chanmask, u16 enable);
u16     S626_DIOCapEnableGet        (HBD hbd, u16 group);
u16     S626_DIOCapStatus           (HBD hbd, u16 group);
void    S626_DIOCapReset            (HBD hbd, u16 group, u16 value);
u16     S626_DIOIntEnableGet        (HBD hbd, u16 group);
void    S626_DIOIntEnableSet        (HBD hbd, u16 group, u16 value);
u16     S626_DIOModeGet             (HBD hbd, u16 group);
void    S626_DIOModeSet             (HBD hbd, u16 group, u16 value);

// Counter
ERR826  S626_CounterLatchSourceSet  (HBD hbd, u16 chan, u16 value);
ERR826  S626_CounterSoftIndex       (HBD hbd, u16 chan);
ERR826  S626_CounterPreload         (HBD hbd, u16 chan, uint data);
ERR826  S626_CounterReadLatch       (HBD hbd, u16 chan, uint *counts);

void    S626_CounterIntSourceSet    (HBD hbd, u16 chan, u16 intsrcEnum);
void    S626_CounterModeSet         (HBD hbd, u16 chan, u16 mode);
u16     S626_CounterModeGet         (HBD hbd, u16 chan);
ERR826  S626_CounterEnableSet       (HBD hbd, u16 chan, u16 enable);
void    S626_CounterLoadTrigSet     (HBD hbd, u16 chan, u16 value);
u16     S626_CounterCapStatus       (HBD hbd);
void    S626_CounterCapFlagsReset   (HBD hbd, u16 chan);

// Battery
u16     S626_BackupEnableGet        (HBD hbd);
void    S626_BackupEnableSet        (HBD hbd, u16 en);
u16     S626_ChargeEnableGet        (HBD hbd);
void    S626_ChargeEnableSet        (HBD hbd, u16 en);

// Watchdog
ERR826  S626_WatchdogTimeout        (HBD hbd, u16 *timeout);
ERR826  S626_WatchdogEnableGet      (HBD hbd, uint *enable);
ERR826  S626_WatchdogEnableSet      (HBD hbd, uint enable);
ERR826  S626_WatchdogPeriodGet      (HBD hbd, u16 *periodEnum);
ERR826  S626_WatchdogPeriodSet      (HBD hbd, u16 periodEnum);
ERR826  S626_WatchdogReset          (HBD hbd);

#ifdef __cplusplus
}
#endif

#endif // #ifdef _INC_PORT626_H_
