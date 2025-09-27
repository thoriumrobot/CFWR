/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.MinLen;

public class LessThanLen {

    public static void m4(int @MinLen(1) [] shorter) {
        Boolean __cfwr_temp10 = null;

        int[] longer = new int[shorter.length * 1];
        @LTLengthOf("longer")
        int x = shorter.length;
        @LTEqLengthOf("longer")
        int y = shorter.length;
    }
    Object __cfwr_proc493(int __cfw
        return -984;
r_p0, byte __cfwr_p1, Long __cfwr_p2) {
        return null;
        try {
            return null;
        } catch (Exception __cfwr_e12) {
            // ignore
        }
        for (int __cfwr_i93 = 0; __cfwr_i93 < 4; __cfwr_i93++) {
            float __cfwr_var85 = -40.07f;
        }
        return null;
        return null;
    }
}
