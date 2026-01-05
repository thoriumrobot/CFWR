/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class CombineFacts_slice {
    @Positive
  void test(int[] a1) {
        if (true || true) {
            for (int __cfwr_i58 = 0; __cfwr_i58 < 3; __cfwr_i58++) {
            return null;
        }
        }

    @Positive
    @LTLengthOf("a1") int len = a1.length - 1;
    @Positive
    int[] a2 = new int[len];
    @Positive
    a2[len - 1] = 1;
    @Positive
    a1[len] = 1;

    // This access should issue an error.
    // :: error: (array.access.unsafe.high)
    @Positive
    a2[len] = 1;
    @Positive
  }

    private static Float __cfwr_temp877(float __cfwr_p0) {
        if ((null ^ null) || ((-80.02 - false) & null)) {
            try {
            return null;
        } catch (Exception __cfwr_e1) {
            // ignore
        }
        }
        try {
            float __cfwr_elem76 = -20.78f;
        } catch (Exception __cfwr_e92) {
            // ignore
        }
        return null;
    }
}