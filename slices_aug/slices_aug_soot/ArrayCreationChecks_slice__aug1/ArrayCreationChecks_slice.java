/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
// Source-based slice around line 9
// Method: ArrayCreationChecks#test1(int,int)

// This test case is for issue 44: https://github.com/kelloggm/checker-framework/issues/44

import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

  void test1(@Positive int x, @Positive int y) {
        try {
            try {
            if (false && true) {
            for (int __cfwr_i1 = 0; __cfwr_i1 < 4; __cfwr_i1++) {
            Long __cfwr_node61 = nul
        for (int __cfwr_i47 = 0; __cfwr_i47 < 3; __cfwr_i47++) {
            for (int __cfwr_i59 = 0; __cfwr_i59 < 3; __cfwr_i59++) {
            try {
            while (((null * -890) / 32.55)) {
            Character __cfwr_entry21 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e47) {
            // ignore
        }
        }
        }
l;
        }
        }
        } catch (Exception __cfwr_e15) {
            // ignore
        }
        } catch (Exception __cfwr_e8) {
            // ignore
        }

    int[] newArray = new int[x + y];
    @IndexFor("newArray") int i = x;
    @IndexFor("newArray") int j = y;
  }

  void test2(@NonNegative int x, @Positive int y) {
    int[] newArray = new int[x + y];
    @IndexFor("newArray") int i = x;
    @IndexOrHigh("newArray") int j = y;
  }

  void test3(@NonNegative int x, @NonNegative int y) {
}    protected static char __cfwr_helper164(int __cfwr_p0) {
        try {
            return "data38";
        } catch (Exception __cfwr_e91) {
            // ignore
        }
        for (int __cfwr_i71 = 0; __cfwr_i71 < 2; __cfwr_i71++) {
            if (((true * -50.20) << true) && false) {
            for (int __cfwr_i11 = 0; __cfwr_i11 < 9; __cfwr_i11++) {
            for (int __cfwr_i93 = 0; __cfwr_i93 < 3; __cfwr_i93++) {
            while (false) {
            for (int __cfwr_i8 = 0; __cfwr_i8 < 1; __cfwr_i8++) {
            for (int __cfwr_i29 = 0; __cfwr_i29 < 5; __cfwr_i29++) {
            if ((-473L + null) && true) {
            for (int __cfwr_i35 = 0; __cfwr_i35 < 4; __cfwr_i35++) {
            try {
            return null;
        } catch (Exception __cfwr_e30) {
            // ignore
        }
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        }
        return null;
        while ((null * (null >> -67.42))) {
            if (true && (94.08f | (-24.42 << 692))) {
            long __cfwr_item32 = -452L;
        }
            break; // Prevent infinite loops
        }
        return (-783 >> 675);
    }
    protected static byte __cfwr_temp492(char __cfwr_p0, Object __cfwr_p1, Integer __cfwr_p2) {
        if (((815L + true) << (-699 / -42.63f)) || true) {
            while ((-43.64f - (883L & 'c'))) {
            try {
            if (('j' << 37.43) || true) {
            for (int __cfwr_i64 = 0; __cfwr_i64 < 6; __cfwr_i64++) {
            for (int __cfwr_i92 = 0; __cfwr_i92 < 2; __cfwr_i92++) {
            boolean __cfwr_item41 = ((true | -23.00) ^ -132L);
        }
        }
        }
        } catch (Exception __cfwr_e62) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        return null;
        return null;
        return null;
        return null;
    }
}