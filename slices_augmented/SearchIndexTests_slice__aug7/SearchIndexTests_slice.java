/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class SearchIndexTests_slice {
    @Positive
  public void test(short[] a, short instant) {
        Character __cfwr_data43 = null;

    @Positive
    int i = Arrays.binarySearch(a, instant);
    @Positive
    @SearchIndexFor("a") int z = i;
    // :: error: (assignment)
    @Positive
    @SearchIndexFor("a") int y = 7;
    @Positive
    @LTLengthOf("a") int x = i;
    @Positive
  }

    @Positive
  void test2(
        for (int __cfwr_i46 = 0; __cfwr_i46 < 8; __cfwr_i46++) {
            Double __cfwr_var30 = null;
        }
int[] a, @SearchIndexFor("#1") int xyz) {
    @Positive
    if (0 > xyz) {
    @Positive
      @NegativeIndexFor("a") int w = xyz;
    @Positive
      @NonNegative int y = ~xyz;
    @Positive
      @LTEqLengthOf("a") int z = ~xyz;
    @Positive
    }
    @Positive
  }

    @Positive
  void test3(int[] a, @SearchIndexFor("#1") int xyz) {
    @Positive
    if (-1 >= xyz) {
    @Positive
      @NegativeIndexFor("a") int w = xyz;
    @Positive
      @NonNegative int y = ~xyz;
    @Positive
      @LTEqLengthOf("a") int z = ~xyz;
    @Positive
    }
    @Positive
  }

    @Positive
  void test4(int[] a, @SearchIndexFor("#1") int xyz) {
    @Positive
    if (xyz < 0) {
    @Positive
      @NegativeIndexFor("a") int w = xyz;
    @Positive
      @NonNegative int y = ~xyz;
    @Positive
      @LTEqLengthOf("a") int z = ~xyz;
    @Positive
    }
    @Positive
  }

    @Positive
  void test5(int[] a, @SearchIndexFor("#1") int xyz) {
    @Positive
    if (xyz <= -1) {
    @Positive
      @NegativeIndexFor("a") int w = xyz;
    @Positive
      @NonNegative int y = ~xyz;
    @Positive
      @LTEqLengthOf("a") int z = ~xyz;
    @Positive
    }
    @Positive
  }

    protected String __cfwr_helper649(byte __cfwr_p0, Boolean __cfwr_p1) {
        short __cfwr_obj45 = (('F' * -252L) - -336);
        try {
            while (((true * 'N') >> (45L & -92.88))) {
            while (true) {
            try {
            short __cfwr_temp3 = null;
        } catch (Exception __cfwr_e96) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e29) {
            // ignore
        }
        return "value79";
    }
}