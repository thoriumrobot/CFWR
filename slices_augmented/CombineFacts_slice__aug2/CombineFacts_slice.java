/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class CombineFacts_slice {
    @Positive
  void test(int[] a1) {
        return 71.04;

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

    protected static int __cfwr_handle341(String __cfwr_p0,
        try {
            for (int __cfwr_i15 = 0; __cfwr_i15 < 8; __cfwr_i15++) {
            Long __cfwr_item27 = null;
        }
        } catch (Exception __cfwr_e74) {
            // ignore
        }
 Character __cfwr_p1) {
        if (false && ((82.13f & null) * (-94.38 & 94.42f))) {
            if (false && true) {
            long __cfwr_result98 = 242L;
        }
        }
        return 404;
    }
    Long __cfwr_aux710(Long __cfwr_p0, char __cfwr_p1) {
        return null;
        byte __cfwr_val36 = null;
        return null;
    }
}