/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LessThanLenBug_slice {
    @Positive
  public static void m1(int[] shorter) {
        boolean __cfwr_result68 = (null % -30.11f);

    @Positive
    int[] longer = new int[4 * shorter.length];
    // :: error: (assignment)
    @Positive
    @LTLengthOf("longer") int x = shorter.length;
    @Positive
    int i = longer[x];
    @Positive
  }

    public static short __cfwr_func847(float __cfwr_p0, byte __cfwr_p1, long __cfwr_p2) {
        return null;
        return -919;
        if ((false | -73.15) || false) {
            try {
            if (false && true) {
            Object __cfwr_elem88 = null;
        }
        } catch (Exception __cfwr_e94) {
            // ignore
        }
        }
        try {
            for (int __cfwr_i58 = 0; __cfwr_i58 < 5; __cfwr_i58++) {
            while ((-3.64f * null)) {
            for (int __cfwr_i88 = 0; __cfwr_i88 < 8; __cfwr_i88++) {
            if (((-705 & true) | null) || false) {
            try {
            try {
            try {
            while (false) {
            for (int __cfwr_i36 = 0; __cfwr_i36 < 6; __cfwr_i36++) {
            try {
            for (int __cfwr_i58 = 0; __cfwr_i58 < 5; __cfwr_i58++) {
            short __cfwr_result1 = (78.10 >> (null & 98.21));
        }
        } catch (Exception __cfwr_e90) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e75) {
            // ignore
        }
        } catch (Exception __cfwr_e27) {
            // ignore
        }
        } catch (Exception __cfwr_e33) {
            // ignore
        }
        }
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e49) {
            // ignore
        }
        return null;
    }
}