/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class CustomContractWithArgs_slice {
    @Positive
    void minLenUse(int[] b) {
        try {
            for (int __cfwr_i89 = 0; __cfwr_i89 < 8; __cfwr_i89++) {
            return null;
        }
        } catch (Exception __cfwr_e60) {
            // ignore
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
 
        if ((78L % -8) && false) {
            Object __cfwr_temp24 = null;
        }
   @Positive
        targetValue = {"#1", "#1"},
    @Positive
        targetOffset = {"#2 + 1", "10"},
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

    private Double __cfwr_process321(Object __cfwr_p0, Object __cfwr_p1, double __cfwr_p2) {
        if (true || (false << 87.40)) {
            try {
            for (int __cfwr_i65 = 0; __cfwr_i65 < 9; __cfwr_i65++) {
            return -29.39f;
        }
        } catch (Exception __cfwr_e8) {
            // ignore
        }
        }
        while (true) {
            try {
            return true;
        } catch (Exception __cfwr_e34) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}