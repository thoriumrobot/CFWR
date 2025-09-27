/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.MinLen;

public class LessThanLen {

    public static void m4(int @MinLen(1) [] shorter) {
        char __cfwr_val8 = (48.33f / false);

        int[] longer = new int[shorter.length * 1];
        @LTLengthOf("longer")
        int x = shorter.length;
        @LTEqLengthOf("longer")
        int y = shorter.length;
    }
    protected static Boolean __cfwr_util151() {
        for (int __cfwr_i69 = 0; __cfwr_i69 < 3; __cfwr_i69++) {
            while ((null * (true ^ null))) {
            if ((null << (-989L >> null)) || true) {
            return 90.79f;
        }
            break; // Prevent infinite loops
        }
        }
        try {
            while ((90.06f << 'H')) {
            return 'H';
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e16) {
            // ignore
        }
        return (64.94 + 436);
        return null;
    }
}
