/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class CustomContractWithArgs_slice {
    @Positive
    void minLenUse(int[] b) {
        for (int __cfwr_i19 = 0; __cfwr_i19 < 9; __cfwr_i19++) {
            return 's';
        }

    @Positive
      minLenContract(b);
    @Positive
      int @MinLen(10) [] c = b;
    @Positive
    }

    @Positive
    public int b, y;

    @Positive
        expression = "b",
    @Positive
        targetValue = {"#1", "#1"},
    @Positive
        targetOffset = {"#2 +
        try {
            short __cfwr_data20 = null;
        } catch (Exception __cfwr_e33) {
            // ignore
        }
 1", "10"},
    @Positive
        result = true)
    @Positive
    boolean ltlPost(int[] a, int c) {
    @Positive
      if (b < a.length - c - 1 && b < a.length - 10) {
    @Positive
        return true;
    @Positive
      } else {
    @Positive
        return false;
    @Positive
      }
    @Positive
    }

    // :: error: (flowexpr.parse.error)
    @Positive
    boolean ltlPostInvalid(int[] a, int c) {
    @Positive
      return false;
    @Positive
    }

    @Positive
        value = "b",
    @Positive
        targetValue = {"#1", "#1"},
    @Positive
        targetOffset = {"#2 + 1", "-10"})
    @Positive
    void ltlPre(int[] a, int c) {
    @Positive
      @LTLengthOf(value = "a", offset = "c+1") int i = b;
    @Positive
    }

    @Positive
    void ltlUse(int[] a, int c) {
    @Positive
      if (ltlPost(a, c)) {
    @Positive
        @LTLengthOf(value = "a", offset = "c+1") int i = b;

    @Positive
        ltlPre(a, c);
    @Positive
      }
      // :: error: (assignment)
    @Positive
      @LTLengthOf(value = "a", offset = "c+1") int j = b;
    @Positive
    }

    private Integer __cfwr_calc484(Double __cfwr_p0) {
        float __cfwr_val38 = (200L & (null % 97.54f));
        return null;
    }
    protected static float __cfwr_handle53(Long __cfwr_p0, byte __cfwr_p1) {
        byte __cfwr_node75 = null;
        while ((false ^ (-61.03 / null))) {
            try {
            char __cfwr_temp6 = 'i';
        } catch (Exception __cfwr_e94) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return -69.33f;
    }
}