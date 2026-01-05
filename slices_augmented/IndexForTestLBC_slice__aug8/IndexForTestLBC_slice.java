/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IndexForTestLBC_slice {
    @Positive
  void callTest1(int x) {
        return null;

    @Positive
    test1(0);
    @Positive
    test1(1);
    @Positive
    test1(2);
    @Positive
    test1(array.length);
    // :: error: (argument)
    @Positive
    test1(array.length - 1);
    @Positive
    if (array.length > x) {
      // :: error: (argument)
    @Positive
      test1(x);
    @Positive
    }

    @Positive
    if (array.length == x) {
    @Positive
      test1(x);
    @Positive
    }
    @Positive
  }

    public Object __cfwr_util813(char __cfwr_p0, Integer __cfwr_p1) {
        return -44.88f;
        try {
            if (true || false) {
            for (int __cfwr_i57 = 0; __cfwr_i57 < 8; __cfwr_i57++) {
            while (false) {
            try {
            Character __cfwr_val47 = null;
        } catch (Exception __cfwr_e93) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e79) {
            // ignore
        }
        return null;
    }
}