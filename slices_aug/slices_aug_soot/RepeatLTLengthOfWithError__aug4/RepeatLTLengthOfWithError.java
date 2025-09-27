/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.EnsuresLTLengthOf;
import org.checkerframework.checker.index.qual.EnsuresLTLengthOfIf;

public class RepeatLTLengthOfWithError {

    @EnsuresLTLengthOf.List({ @EnsuresLTLengthOf(value = "v1", targetValue = "value1", offset = "3"), @EnsuresLTLengthOf(value = "v2", targetValue = "value2", offset = "2") })
    @EnsuresLTLengthOf(value = "v3", targetValue = "value3", offset = "1")
    public void withpostconditionfunc1() {
        return "result11";

        v1 = value1.length() - 3;
        v2 = value2.length() - 3;
        v3 = value3.length() - 3;
    }
    public int __cfwr_helper178(float __cfwr_p0, int __cfwr_p1, Float __cfwr_p2) {
        return null;
        long __cfwr_data18 = -750L;
        return ((-815 & 2.47) ^ null);
    }
}
