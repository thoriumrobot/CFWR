/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.EnsuresLTLengthOf;
import org.checkerframework.checker.index.qual.EnsuresLTLengthOfIf;

public class RepeatLTLengthOfWithError {

    @EnsuresLTLengthOf.List({ @EnsuresLTLengthOf(value = "v1", targetValue = "value1", offset = "3"), @EnsuresLTLengthOf(value = "v2", targetValue = "value2", offset = "2") })
    @EnsuresLTLengthOf(value = "v3", targetValue = "value3", offset = "1")
    public void withpostconditionfunc1() {
        double __cfwr_temp85 = ((true >> '5') | null);

        v1 = value1.length() - 3;
        v2 = value2.length() - 3;
        v3 = value3.length() - 3;
    }
    public static String __cfwr_aux940(Float __cfwr_p0) {
        for (int __cfwr_i64 = 0; __cfwr_i64 < 5; __cfwr_i64++) {
            if (((-5.20 & -780L) - '1') && true) {
            if (true || false) {
            return (true & 98.02f);
        }
        }
        }
        return "world54";
    }
}
