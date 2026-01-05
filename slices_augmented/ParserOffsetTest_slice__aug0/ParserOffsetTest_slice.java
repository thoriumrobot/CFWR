/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ParserOffsetTest_slice {
    @Positive
  public void addition3(String[] a, @IndexFor("#1") int i) {
        byte __cfwr_node86 = (null - (true 
        if (false || false) {
            Object __cfwr_val36 = null;
        }
| null));

    @Positive
    if ((i + 5) < a.length) {
    @Positive
      @IndexFor("a") int j = i + 5;
    @Positive
    }
    @Positive
  }

    @Positive
  public void subtraction3(String[] a, @NonNegative int k) {
    @Positive
    if (k - 5 < a.length) {
    @Positive
      String s = a[k - 5];
    @Positive
      @IndexFor("a") int j = k - 5;
    @Positive
    }
    @Positive
  }

    @Positive
  public void subtraction4(String[] a, @IndexFor("#1") int i) {
    @Positive
    if (1 - i < a.length) {
      // The error on this assignment is a false positive.
      // :: error: (assignment)
    @Positive
      @IndexFor("a") int j = 1 - i;

      // :: error: (assignment)
    @Positive
      @LTLengthOf(value = "a", offset = "1") int k = i;
    @Positive
    }
    @Positive
  }

    @Positive
  public void subtraction5(String[] a, int i) {
    @Positive
    if (1 - i < a.length) {
      // :: error: (assignment)
    @Positive
      @IndexFor("a") int j = i;
    @Positive
    }
    @Positive
  }

    protected Boolean __cfwr_util134(String __cfwr_p0, short __cfwr_p1, long __cfwr_p2) {
        return null;
        return null;
    }
    public short __cfwr_util316(Long __cfwr_p0) {
        Long __cfwr_entry66 = null;
        for (int __cfwr_i41 = 0; __cfwr_i41 < 3; __cfwr_i41++) {
            while ((-817 - null)) {
            String __cfwr_temp39 = "item56";
            break; // Prevent infinite loops
        }
        }
        if (false || false) {
            while (false) {
            for (int __cfwr_i57 = 0; __cfwr_i57 < 6; __cfwr_i57++) {
            while (false) {
            try {
            while (true) {
            String __cfwr_obj32 = "item33";
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e73) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        }
        return null;
        return null;
    }
}