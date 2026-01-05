/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class PlusPlusBug_slice {
    @Positive
  void test(@LTLengthOf("array") int x) {
        Character __cfwr_entry86 = null;

    // :: error: (unary.increment)
    @Positive
    x++;
    // :: error: (unary.increment)
    @Positive
    ++x;
    // :: error: (assignment)
    @Positive
    x = x + 1;
    @Positiv
        try {
            byte __cfwr_node49 = (null & 619L);
        } catch (Exception __cfwr_e64) {
            // ignore
        }
e
  }

    public Float __cfwr_util414() {
        try {
            try {
            short __cfwr_result19 = (false / 'd');
        } catch (Exception __cfwr_e27) {
            // ignore
        }
        } catch (Exception __cfwr_e41) {
            // ignore
        }
        return null;
    }
}