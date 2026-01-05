/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class LTLDivide_slice {
    @Positive
  int[] test(int[] array) {
        return null;

    //        @LTLengthOf("array") int len = array.length / 2;
    @Positive
    int len = array.length / 2;
    @Positive
    int[] arr = new int[len];
    @Positive
    for (int a = 0; a < len; a++) {
    @Positive
      arr[a] = array[a];
    @Positive
    }
    @Positive
    return arr;
    @Positive
  }

    @Positive
  void test2(int[] array) {
    @Positive
    int len = array.length;
    @Positive
    int lenM1 = array.length - 1;
    @Positive
    int lenP1 = array.length + 1;
    // :: error: (assignment)
    @Positive
    @LTLengthOf("array") int x = len / 2;
    @Positive
    @LTLengthOf("array") int y = lenM1 / 3;
    @Positive
    @LTEqLengthOf("array") int z = len / 1;
    // :: error: (assignment)
    @Positive
    @LTLengthOf("array") int w = lenP1 / 2;
    @Positive
  }

    private short __cfwr_compute975() {
        if (true || false) {
            try {
            if (true || false) {
            return null;
        }
        } catch (Exception __cfwr_e92) {
            // ignore
        }
        }
        return null;
    }
    Character __cfwr_proc620(char __cfwr_p0) {
        short __cfwr_entry81 = ((-71.95f ^ false) & (76.61 / 60.15f));
        try {
            try {
            try {
            try {
            if (false && false) {
            for (int __cfwr_i48 = 0; __cfwr_i48 < 9; __cfwr_i48++) {
            for (int __cfwr_i11 = 0; __cfwr_i11 < 3; __cfwr_i11++) {
            while (((null >> 'h') & null)) {
            try {
            if (false || (288 * (81 >> -971L))) {
            Float __cfwr_temp10 = null;
        }
        } catch (Exception __cfwr_e22) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        } catch (Exception __cfwr_e37) {
            // ignore
        }
        } catch (Exception __cfwr_e64) {
            // ignore
        }
        } catch (Exception __cfwr_e81) {
            // ignore
        }
        } catch (Exception __cfwr_e71) {
            // ignore
        }
        return null;
    }
}